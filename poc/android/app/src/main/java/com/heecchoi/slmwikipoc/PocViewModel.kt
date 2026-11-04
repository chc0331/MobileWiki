package com.heecchoi.slmwikipoc

import android.app.Application
import android.app.ActivityManager
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.Environment
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID
import kotlin.math.ceil

enum class AppTab(val label: String) { NOTES("문서"), GROUPS("군집"), SEARCH("검색"), EVALUATION("평가") }

data class PerformanceResult(
    val coldP95Ms: Double,
    val warmP95Ms: Double,
    val coldFailures: Int,
    val warmFailures: Int,
    val maxAdditionalPssMb: Double,
    val modelFilesBytes: Long,
    val measuredAt: Long,
)

data class DeviceSummary(val manufacturer: String, val model: String, val androidRelease: String, val sdk: Int, val abi: String?, val hardware: String, val ramBytes: Long, val cpuCores: Int) {
    val meetsPocTarget: Boolean get() = sdk >= 31 && abi?.contains("arm64") == true && ramBytes >= 8_000_000_000L
}

data class AppUiState(
    val modelStatus: String = "학습 모델을 확인하는 중입니다",
    val modelReady: Boolean = false,
    val modelError: String? = null,
    val tab: AppTab = AppTab.NOTES,
    val workspace: String = "experiment",
    val clusterThreshold: Double = 0.05,
    val experimentThreshold: Double = 0.05,
    val device: DeviceSummary? = null,
    val notes: List<Note> = emptyList(),
    val groups: List<NoteGroup> = emptyList(),
    val groupCalculationActive: Boolean = false,
    val projectedDocuments: List<ProjectedDocument> = emptyList(),
    val experimentMetrics: ExperimentClusteringMetrics? = null,
    val importPreview: List<ImportPreviewItem>? = null,
    val scanActive: Boolean = false,
    val scanProgress: Pair<Int, Int>? = null,
    val scanReport: String? = null,
    val query: String = "",
    val searchMode: SearchMode = SearchMode.HYBRID,
    val hits: List<SearchHit> = emptyList(),
    val loading: Boolean = false,
    val busyMessage: String? = null,
    val message: String? = null,
    val evaluation: EvaluationResult? = null,
    val evaluationProgress: Pair<Int, Int>? = null,
    val performance: PerformanceResult? = null,
    val dailySummaries: List<DailyUsageSummary> = emptyList(),
)

class PocViewModel(application: Application) : AndroidViewModel(application) {
    private val store = LocalStore(application)
    private val model: EmbeddingModel = EmbeddingModel.from(application)
    private var experimentProjectionKey: List<Pair<String, Long>>? = null
    private var experimentProjectionCache: List<ProjectedDocument> = emptyList()
    private var scanJob: Job? = null
    private var groupJob: Job? = null
    private val initialPssKb = Debug.getPss()
    private val deviceSummary = run {
        val memory = ActivityManager.MemoryInfo()
        (application.getSystemService(Application.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(memory)
        DeviceSummary(Build.MANUFACTURER, Build.MODEL, Build.VERSION.RELEASE, Build.VERSION.SDK_INT, Build.SUPPORTED_ABIS.firstOrNull(), Build.HARDWARE, memory.totalMem, Runtime.getRuntime().availableProcessors())
    }
    private var peakPssKb = initialPssKb
    var state by androidx.compose.runtime.mutableStateOf(AppUiState())
        private set

    init {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { model.load() }
                val manifest = JSONObject(getApplication<Application>().assets.open("evaluation.json").bufferedReader().use { it.readText() }).getJSONObject("manifest")
                val threshold = manifest.getDouble("cluster_threshold")
                state = state.copy(modelReady = true, device = deviceSummary, clusterThreshold = threshold, experimentThreshold = threshold, modelStatus = "학습 어댑터 checkpoint-250 · ONNX Runtime CPU")
                refreshNotes()
                recordPss()
                refreshDailySummaries()
            } catch (error: Throwable) {
                state = state.copy(modelError = error.message ?: error.javaClass.simpleName, modelStatus = "모델을 불러오지 못했습니다")
            }
        }
    }

    fun selectTab(tab: AppTab) {
        state = state.copy(tab = tab, message = null)
        if (tab == AppTab.GROUPS && state.modelReady) refreshGroups()
    }

    fun selectWorkspace(workspace: String) {
        state = state.copy(
            workspace = workspace,
            notes = emptyList(),
            groups = emptyList(),
            groupCalculationActive = false,
            projectedDocuments = emptyList(),
            experimentMetrics = null,
            hits = emptyList(),
            query = "",
            message = null,
        )
        refreshNotes()
    }

    fun setQuery(query: String) { state = state.copy(query = query) }
    fun setSearchMode(mode: SearchMode) { state = state.copy(searchMode = mode) }

    fun saveNote(id: String?, title: String, text: String) {
        val cleanTitle = title.trim()
        val cleanText = text.trim()
        if (cleanTitle.isBlank() || cleanText.isBlank()) {
            state = state.copy(message = "제목과 내용을 모두 입력해 주세요.")
            return
        }
        val existingId = id ?: UUID.randomUUID().toString()
        state = state.copy(loading = true, message = null)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val vectors = model.embedNote(cleanTitle, cleanText)
                    store.upsert(NoteInput(existingId, null, cleanTitle, cleanText), vectors.searchChunks, vectors.cluster)
                }
                state = state.copy(loading = false, message = if (id == null) "메모를 저장했습니다." else "메모를 수정했습니다.")
                refreshNotes()
                if (state.tab == AppTab.GROUPS) refreshGroups()
            } catch (error: Throwable) {
                state = state.copy(loading = false, message = "메모를 저장하지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun deleteNote(note: Note) {
        if (note.workspace !in setOf("personal", "experiment")) return
        viewModelScope.launch(Dispatchers.IO) {
            store.delete(note.id)
            withContext(Dispatchers.Main) {
                state = state.copy(message = "메모를 삭제했습니다.")
                refreshNotes()
            }
        }
    }

    fun previewDocuments(uris: List<Uri>) {
        if (state.workspace != "experiment" || !state.modelReady || uris.isEmpty()) return
        state = state.copy(loading = true, busyMessage = "선택한 문서를 확인하는 중입니다", message = null)
        viewModelScope.launch {
            val previews = mutableListOf<ImportPreviewItem>()
            uris.forEachIndexed { index, uri ->
                state = state.copy(busyMessage = "문서 확인 ${index + 1}/${uris.size}")
                val preview = withContext(Dispatchers.IO) {
                    try {
                        val document = DocumentImportParser.parse(getApplication(), uri)
                        ImportPreviewItem(document, document.fileName)
                    } catch (error: CancellationException) {
                        throw error
                    } catch (error: Throwable) {
                        val fallbackName = uri.lastPathSegment?.substringAfterLast('/')?.substringAfterLast(':') ?: "선택한 파일"
                        ImportPreviewItem(null, fallbackName, error.message ?: "파일을 읽지 못했습니다.")
                    }
                }
                previews += preview
            }
            state = state.copy(loading = false, busyMessage = null, importPreview = previews)
            if (previews.none { it.document != null }) state = state.copy(message = "가져올 수 있는 문서가 없습니다.")
        }
    }

    fun setImportGroundTruth(documentId: String, value: String) {
        val previews = state.importPreview ?: return
        state = state.copy(importPreview = previews.map { item ->
            if (item.document?.id == documentId) item.copy(groundTruthGroup = value) else item
        })
    }

    fun cancelDocumentImport() {
        val previews = state.importPreview ?: return
        viewModelScope.launch(Dispatchers.IO) {
            previews.mapNotNull { it.document }.forEach(DocumentImportParser::cleanup)
        }
        state = state.copy(importPreview = null)
    }

    fun deviceScanPermissionDenied() {
        state = state.copy(message = "모든 파일 접근 권한이 없어 기기 문서를 스캔하지 않았습니다. 수동 문서 가져오기는 사용할 수 있습니다.")
    }

    fun cancelDeviceScan() { scanJob?.cancel() }

    fun startDeviceScan() {
        if (state.workspace != "experiment" || !state.modelReady || state.loading || scanJob?.isActive == true) return
        if (!Environment.isExternalStorageManager()) { deviceScanPermissionDenied(); return }
        groupJob?.cancel()
        state = state.copy(loading = true, scanActive = true, scanProgress = null, scanReport = null, groupCalculationActive = false, busyMessage = "공유 저장소에서 문서를 찾는 중입니다", message = null)
        scanJob = viewModelScope.launch {
            var added = 0
            var updated = 0
            var reused = 0
            var removed = 0
            var pathErrors = 0
            val excluded = mutableMapOf<String, Int>()
            try {
                val discovery = DeviceDocumentScanner.discover(getApplication())
                if (discovery.availableVolumes == 0) error("읽을 수 있는 공유 저장소가 없습니다.")
                val previous = withContext(Dispatchers.IO) { store.scannedDocuments().associateBy { it.path } }
                val seen = HashSet<String>()
                val total = discovery.files.size
                discovery.files.forEachIndexed { index, file ->
                    currentCoroutineContext().ensureActive()
                    state = state.copy(scanProgress = index to total, busyMessage = "문서 확인·임베딩 ${index + 1}/$total: ${file.name}")
                    val path = try { file.canonicalPath } catch (_: Exception) {
                        excluded.merge("경로를 확인할 수 없음", 1, Int::plus)
                        pathErrors++
                        return@forEachIndexed
                    }
                    seen += path
                    val size = file.length()
                    val modifiedAt = file.lastModified()
                    val old = previous[path]
                    if (size > DocumentImportParser.MAX_FILE_BYTES) {
                        excluded.merge("10MB 초과", 1, Int::plus)
                        if (old != null) withContext(Dispatchers.IO) { store.delete(old.id) }
                        return@forEachIndexed
                    }
                    if (old != null && old.size == size && old.modifiedAt == modifiedAt) {
                        reused++
                        return@forEachIndexed
                    }
                    var parsed: ParsedDocument? = null
                    try {
                        withContext(Dispatchers.IO) {
                            val document = DocumentImportParser.parseFile(getApplication(), file)
                            parsed = document
                            check(file.length() == size && file.lastModified() == modifiedAt) { "읽는 동안 파일이 변경됨" }
                            val vectors = model.embedNote(document.title, document.text)
                            val source = DocumentSource(document.fileName, document.mimeType, pageTexts = document.pageTexts, devicePath = path, deviceSize = size, deviceModifiedAt = modifiedAt)
                            val id = old?.id ?: "scan-${UUID.nameUUIDFromBytes(path.toByteArray(Charsets.UTF_8))}"
                            store.upsert(NoteInput(id, null, document.title, document.text, source = source), vectors.searchChunks, vectors.cluster, "experiment")
                        }
                        if (old == null) added++ else updated++
                    } catch (error: CancellationException) {
                        throw error
                    } catch (error: Exception) {
                        val reason = error.message?.substringBefore(':')?.take(60) ?: "읽기 또는 임베딩 실패"
                        excluded.merge(reason, 1, Int::plus)
                    } finally {
                        parsed?.let(DocumentImportParser::cleanup)
                    }
                }
                val complete = discovery.complete && pathErrors == 0
                if (complete) {
                    removed = withContext(Dispatchers.IO) {
                        previous.values.filter { it.path !in seen }.onEach { store.delete(it.id) }.size
                    }
                }
                state = state.copy(scanProgress = total to total, busyMessage = "군집과 지도를 계산하는 중입니다")
                val notes = withContext(Dispatchers.IO) { store.notes("experiment") }
                val threshold = state.experimentThreshold
                val groups = withContext(Dispatchers.Default) {
                    val context = currentCoroutineContext()
                    AverageLinkageGroups.clusterLarge(notes, threshold) { context.ensureActive() }
                }
                val projected = withContext(Dispatchers.Default) { PcaProjector.project(notes) }
                val report = buildString {
                    append("발견 $total · 새 문서 $added · 갱신 $updated · 재사용 $reused · 사라진 문서 정리 $removed")
                    append("\n지원하지 않는 파일 ${discovery.unsupported}개 · 읽을 수 없는 폴더/저장소 ${discovery.unreadableDirectories}개")
                    if (excluded.isNotEmpty()) {
                        append("\n제외/실패: ${excluded.entries.joinToString(" · ") { "${it.key} ${it.value}개" }}")
                        append("\n처리하지 못한 파일이 있어 지도에 이전에 저장한 문서가 남아 있을 수 있습니다.")
                    }
                    if (!complete) append("\n일부 파일에 접근하지 못해 이번 스캔에서는 사라진 문서를 정리하지 않았습니다.")
                }
                experimentProjectionKey = notes.map { it.id to it.updatedAt }
                experimentProjectionCache = projected
                state = state.copy(loading = false, scanActive = false, scanProgress = null, scanReport = report, groupCalculationActive = false, busyMessage = null,
                    message = if (complete && excluded.isEmpty()) "기기 문서 스캔을 마쳤습니다." else "스캔 결과에 제외하거나 읽지 못한 항목이 있습니다.",
                    tab = if (state.workspace == "experiment") AppTab.GROUPS else state.tab,
                    notes = if (state.workspace == "experiment") notes else state.notes,
                    groups = if (state.workspace == "experiment") groups else state.groups,
                    projectedDocuments = if (state.workspace == "experiment") projected else state.projectedDocuments,
                    experimentMetrics = if (state.workspace == "experiment") ExperimentClusteringMetricsCalculator.calculate(notes, groups) else state.experimentMetrics)
            } catch (error: CancellationException) {
                state = state.copy(loading = false, scanActive = false, scanProgress = null,
                    scanReport = "스캔 중단 · 새 문서 $added · 갱신 $updated · 재사용 $reused. 다음 스캔에서 이어서 처리합니다.",
                    busyMessage = null, message = "스캔을 중단했습니다. 일부 문서만 저장되었을 수 있습니다.")
                if (state.workspace == "experiment") refreshNotes()
            } catch (error: Throwable) {
                state = state.copy(loading = false, scanActive = false, scanProgress = null,
                    scanReport = "군집 결과를 완료하지 못했습니다. 새 문서 $added · 갱신 $updated · 재사용 $reused. 다시 스캔하면 저장된 임베딩을 재사용합니다.",
                    groups = emptyList(), projectedDocuments = emptyList(), experimentMetrics = null,
                    busyMessage = null, message = "스캔 또는 군집 계산을 완료하지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
                if (state.workspace == "experiment") refreshNotes()
            } finally {
                scanJob = null
            }
        }
    }

    fun commitDocumentImport() {
        val previews = state.importPreview ?: return
        val documents = previews.mapNotNull { item -> item.document?.let { item to it } }
        if (documents.isEmpty()) return
        state = state.copy(loading = true, busyMessage = "문서를 임베딩하고 저장하는 중입니다", message = null)
        viewModelScope.launch {
            var saved = 0
            val failures = mutableListOf<String>()
            documents.forEachIndexed { index, (item, document) ->
                state = state.copy(busyMessage = "문서 임베딩 ${index + 1}/${documents.size}: ${document.fileName}")
                try {
                    withContext(Dispatchers.IO) {
                        val vectors = model.embedNote(document.title, document.text)
                        val source = DocumentImportParser.persistSource(
                            document,
                            File(getApplication<Application>().noBackupFilesDir, "experiment-documents"),
                        )
                        try {
                            store.upsert(
                                NoteInput(
                                    document.id,
                                    null,
                                    document.title,
                                    document.text,
                                    item.groundTruthGroup.trim().takeIf(String::isNotEmpty),
                                    source,
                                ),
                                vectors.searchChunks,
                                vectors.cluster,
                                "experiment",
                            )
                        } catch (error: Throwable) {
                            source.storedPath?.let { File(it).delete() }
                            throw error
                        }
                    }
                    saved++
                } catch (error: CancellationException) {
                    throw error
                } catch (error: Throwable) {
                    failures += "${document.fileName}: ${error.message ?: error.javaClass.simpleName}"
                } finally {
                    DocumentImportParser.cleanup(document)
                }
            }
            state = state.copy(
                loading = false,
                busyMessage = null,
                importPreview = null,
                message = buildString {
                    append("${saved}개 문서를 군집 실험 공간에 추가했습니다.")
                    if (failures.isNotEmpty()) append(" 실패 ${failures.size}개: ${failures.joinToString(" · ")}")
                },
            )
            refreshNotes()
        }
    }

    fun updateExperimentGroundTruth(note: Note, value: String) {
        if (note.workspace != "experiment") return
        viewModelScope.launch(Dispatchers.IO) {
            store.updateGroundTruthGroup(note.id, value.trim().takeIf(String::isNotEmpty))
            withContext(Dispatchers.Main) {
                state = state.copy(message = "정답 주제를 저장했습니다.")
                refreshNotes()
            }
        }
    }

    fun importSamples() {
        if (!state.modelReady) return
        state = state.copy(loading = true, busyMessage = "합성 샘플 메모를 기기에 추가하는 중입니다", message = null)
        viewModelScope.launch {
            try {
                val added = withContext(Dispatchers.IO) {
                    val samples = JSONArray(getApplication<Application>().assets.open("sample_notes.json").bufferedReader().use { it.readText() })
                    var count = 0
                    for (index in 0 until samples.length()) {
                        val row = samples.getJSONObject(index)
                        val id = row.getString("doc_id")
                        if (store.importedSample(id)) continue
                        val title = row.getString("title")
                        val text = row.getString("text")
                        val vectors = model.embedNote(title, text)
                        store.upsert(NoteInput(id, row.getString("event_id"), title, text), vectors.searchChunks, vectors.cluster, "sample")
                        count++
                    }
                    count
                }
                state = state.copy(loading = false, busyMessage = null, message = "${added}개 샘플 메모를 추가했습니다.")
                refreshNotes()
                refreshGroups()
            } catch (error: Throwable) {
                state = state.copy(loading = false, busyMessage = null, message = "샘플을 가져오지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun search() {
        val query = state.query.trim()
        if (query.isBlank()) { state = state.copy(message = "검색어를 입력해 주세요."); return }
        if (!state.modelReady) { state = state.copy(message = "학습 모델이 준비되지 않았습니다."); return }
        val mode = state.searchMode
        val notes = state.notes
        state = state.copy(loading = true, busyMessage = "기기에서 메모를 검색하는 중입니다", message = null)
        viewModelScope.launch {
            try {
                val hits = withContext(Dispatchers.IO) {
                    val queryVector = model.embedQuery(query)
                    NoteSearch.search(notes, query, queryVector, mode)
                }
                state = state.copy(loading = false, busyMessage = null, hits = hits)
                recordPss()
            } catch (error: Throwable) {
                state = state.copy(loading = false, busyMessage = null, message = "검색하지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun refreshGroups() {
        if (state.scanActive) return
        val notes = state.notes
        val workspace = state.workspace
        val isExperiment = workspace == "experiment"
        val threshold = if (isExperiment) state.experimentThreshold else state.clusterThreshold
        if (!state.modelReady) return
        groupJob?.cancel()
        if (isExperiment) state = state.copy(groupCalculationActive = true)
        groupJob = viewModelScope.launch {
            try {
            val groups = withContext(Dispatchers.Default) {
                if (isExperiment) {
                    val context = currentCoroutineContext()
                    AverageLinkageGroups.clusterLarge(notes, threshold) { context.ensureActive() }
                } else AverageLinkageGroups.cluster(notes, threshold)
            }
            val metrics = if (isExperiment) ExperimentClusteringMetricsCalculator.calculate(notes, groups) else null
            val projectionKey = notes.map { it.id to it.updatedAt }
            val projected = if (isExperiment) {
                if (projectionKey == experimentProjectionKey) experimentProjectionCache
                else withContext(Dispatchers.Default) { PcaProjector.project(notes) }.also {
                    experimentProjectionKey = projectionKey
                    experimentProjectionCache = it
                }
            } else emptyList()
            val currentThreshold = if (state.workspace == "experiment") state.experimentThreshold else state.clusterThreshold
            if (state.workspace != workspace || currentThreshold != threshold || state.notes.map { it.id to it.updatedAt } != projectionKey) return@launch
            state = state.copy(groups = groups, experimentMetrics = metrics, projectedDocuments = projected, groupCalculationActive = false)
            } catch (_: CancellationException) {
                // A newer threshold or workspace superseded this calculation.
            } catch (error: Throwable) {
                if (state.workspace == workspace) state = state.copy(message = "군집을 계산하지 못했습니다: ${error.message ?: error.javaClass.simpleName}", groups = emptyList(), projectedDocuments = emptyList(), experimentMetrics = null, groupCalculationActive = false)
            }
        }
    }

    fun setExperimentThreshold(value: Float) {
        state = state.copy(experimentThreshold = value.toDouble().coerceIn(0.0, 2.0), groupCalculationActive = true)
    }

    fun applyExperimentThreshold() {
        if (state.workspace == "experiment") refreshGroups()
    }

    fun feedback(type: String, id: String, positive: Boolean) {
        viewModelScope.launch(Dispatchers.IO) {
            store.saveFeedback(type, id, positive)
            val summaries = store.dailySummaries()
            withContext(Dispatchers.Main) { state = state.copy(dailySummaries = summaries) }
        }
        state = state.copy(message = if (positive) "도움이 됐다고 기록했습니다." else "다시 살펴볼 결과로 기록했습니다.")
    }

    fun runEvaluation() {
        if (!state.modelReady) { state = state.copy(message = "학습 모델이 준비되지 않았습니다."); return }
        state = state.copy(loading = true, busyMessage = "고정 자료의 메모를 임베딩하는 중입니다", evaluationProgress = 0 to 1000, message = null)
        viewModelScope.launch {
            try {
                val result = withContext(Dispatchers.IO) {
                    OnDeviceEvaluation(getApplication(), store, model).run { current, total ->
                        launch(Dispatchers.Main) { state = state.copy(busyMessage = "평가 자료 임베딩 $current/$total", evaluationProgress = current to total) }
                    }
                }
                state = state.copy(loading = false, busyMessage = null, evaluationProgress = null, evaluation = result, message = "고정 평가를 마쳤습니다. 자동 정답은 사람 검수 전 잠정 결과입니다.")
                recordPss()
            } catch (error: Throwable) {
                state = state.copy(loading = false, busyMessage = null, evaluationProgress = null, message = "평가를 마치지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun runPerformanceMeasurement() {
        if (!state.modelReady) { state = state.copy(message = "학습 모델이 준비되지 않았습니다."); return }
        val notes = state.notes
        val query = state.query.ifBlank { "아이 병원 어디로 가기로 했었지?" }
        val mode = state.searchMode
        state = state.copy(loading = true, busyMessage = "콜드 검색 30회와 재사용 검색 100회를 측정합니다", message = null)
        viewModelScope.launch {
            val cold = mutableListOf<Double>()
            val warm = mutableListOf<Double>()
            var coldFailures = 0
            var warmFailures = 0
            try {
                withContext(Dispatchers.IO) {
                    model.close()
                    repeat(30) { index ->
                        val trial = try {
                            val started = System.nanoTime()
                            model.load()
                            val vector = model.embedQuery(query)
                            NoteSearch.search(notes, query, vector, mode)
                            val elapsed = (System.nanoTime() - started) / 1_000_000.0
                            recordPss()
                            model.close()
                            if (elapsed > 2_000.0) { coldFailures++; Double.POSITIVE_INFINITY } else elapsed
                        } catch (error: Throwable) {
                            model.close()
                            coldFailures++
                            Double.POSITIVE_INFINITY
                        }
                        cold += trial
                        launch(Dispatchers.Main) { state = state.copy(busyMessage = "콜드 검색 ${index + 1}/30") }
                    }
                    model.load()
                    repeat(100) { index ->
                        val started = System.nanoTime()
                        try {
                            val vector = model.embedQuery(query)
                            NoteSearch.search(notes, query, vector, mode)
                            val elapsed = (System.nanoTime() - started) / 1_000_000.0
                            if (elapsed > 2_000.0) { warmFailures++; warm += Double.POSITIVE_INFINITY } else warm += elapsed
                            recordPss()
                        } catch (error: Throwable) {
                            warmFailures++
                            warm += Double.POSITIVE_INFINITY
                        }
                        if ((index + 1) % 10 == 0) launch(Dispatchers.Main) { state = state.copy(busyMessage = "재사용 검색 ${index + 1}/100") }
                    }
                }
                val manifest = JSONObject(getApplication<Application>().assets.open("model/embedding_encoder.manifest.json").bufferedReader().use { it.readText() })
                val size = manifest.getLong("onnx_size_bytes") + manifest.getLong("tokenizer_size_bytes")
                val result = PerformanceResult(percentile95(cold), percentile95(warm), coldFailures, warmFailures, ((peakPssKb - initialPssKb).coerceAtLeast(0) / 1024.0), size, System.currentTimeMillis())
                withContext(Dispatchers.IO) { store.savePerformance(result) }
                state = state.copy(
                    loading = false,
                    busyMessage = null,
                    performance = result,
                    dailySummaries = withContext(Dispatchers.IO) { store.dailySummaries() },
                    message = "기기 성능 측정을 마쳤습니다. 2초 초과·실패는 p95 계산에서 시간 초과로 포함했습니다.",
                )
            } catch (error: Throwable) {
                state = state.copy(loading = false, busyMessage = null, message = "성능 측정을 마치지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    fun exportReport(uri: Uri?) {
        if (uri == null) return
        val evaluation = state.evaluation
        val performance = state.performance
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val manifest = JSONObject(getApplication<Application>().assets.open("model/embedding_encoder.manifest.json").bufferedReader().use { it.readText() })
                    val report = JSONObject()
                        .put("created_at_utc_ms", System.currentTimeMillis())
                        .put("app_version", getApplication<Application>().packageManager.getPackageInfo(getApplication<Application>().packageName, 0).versionName)
                        .put("device", JSONObject().put("manufacturer", deviceSummary.manufacturer).put("model", deviceSummary.model).put("android_release", deviceSummary.androidRelease).put("sdk", deviceSummary.sdk).put("abi", deviceSummary.abi).put("hardware", deviceSummary.hardware).put("ram_bytes", deviceSummary.ramBytes).put("available_cpu_cores", deviceSummary.cpuCores))
                        .put("model", JSONObject().put("runtime", manifest.getString("runtime")).put("source_model_repo", manifest.getString("source_model_repo")).put("source_model_revision", manifest.getString("source_model_revision")).put("artifact_repo", manifest.getString("model_hub_repo")).put("artifact_revision", manifest.getString("model_hub_revision")).put("base_model", manifest.getString("base_model")).put("base_revision", manifest.getString("base_revision")).put("adapter_checkpoint", manifest.getString("adapter_checkpoint")).put("precision", manifest.getString("precision")).put("sha256", manifest.getString("onnx_sha256")).put("bytes", manifest.getLong("onnx_size_bytes")).put("tokenizer_sha256", manifest.getString("tokenizer_sha256")).put("tokenizer_bytes", manifest.getLong("tokenizer_size_bytes")))
                        .put("dataset", JSONObject().put("id", "heecchoi/slm-wiki-embeddinggemma2-text-synthetic-v1").put("revision", "ddba19537f56966c5fbe352707b93e71f57ccdf0").put("human_review", "pending; metrics provisional"))
                    if (evaluation != null) {
                        val retrieval = JSONObject()
                        evaluation.retrieval.forEach { (mode, values) -> retrieval.put(mode.name.lowercase(), JSONObject().put("recall_at_1", values.recallAt1).put("recall_at_5", values.recallAt5).put("mrr", values.mrr)) }
                        report.put("evaluation", JSONObject().put("retrieval", retrieval).put("sampled_pair_f1", evaluation.sampledPairs.f1).put("sampled_pair_precision", evaluation.sampledPairs.precision).put("sampled_pair_recall", evaluation.sampledPairs.recall).put("all_event_pair_f1", evaluation.allEventPairs.f1).put("all_event_pair_precision", evaluation.allEventPairs.precision).put("all_event_pair_recall", evaluation.allEventPairs.recall).put("overmerged_pairs", evaluation.allEventPairs.falseMerges).put("documents", evaluation.documentCount).put("queries", evaluation.queryCount).put("sampled_pairs", evaluation.pairCount).put("threshold", evaluation.threshold))
                    }
                    if (performance != null) {
                        report.put("performance", JSONObject().put("cold_search_p95_ms", jsonNumber(performance.coldP95Ms)).put("warm_search_p95_ms", jsonNumber(performance.warmP95Ms)).put("cold_failures", performance.coldFailures).put("warm_failures", performance.warmFailures).put("max_additional_pss_mb", performance.maxAdditionalPssMb).put("model_files_bytes", performance.modelFilesBytes).put("measured_at_ms", performance.measuredAt).put("search_timeout_ms", 2000))
                    }
                    val device = deviceSummary
                    report.put("target_device_check", JSONObject().put("meets_android_12_arm64_8gb_target", device.meetsPocTarget).put("ram_bytes", device.ramBytes).put("cpu_cores", device.cpuCores))
                    report.put("daily_summaries", JSONArray().apply {
                        state.dailySummaries.forEach { day -> put(JSONObject().put("day", day.day).put("positive_feedback", day.positive).put("negative_feedback", day.negative).put("cold_search_p95_ms", day.coldP95Ms?.let(::jsonNumber) ?: JSONObject.NULL).put("warm_search_p95_ms", day.warmP95Ms?.let(::jsonNumber) ?: JSONObject.NULL).put("failed_search_runs", day.failedRuns)) }
                    })
                    getApplication<Application>().contentResolver.openOutputStream(uri)?.use { it.write(report.toString(2).toByteArray()) }
                        ?: error("보고서 파일을 쓸 수 없습니다.")
                }
                state = state.copy(message = "개인 메모와 검색어를 포함하지 않은 로컬 보고서를 저장했습니다.")
            } catch (error: Throwable) {
                state = state.copy(message = "보고서를 저장하지 못했습니다: ${error.message ?: error.javaClass.simpleName}")
            }
        }
    }

    private fun refreshNotes() {
        val workspace = state.workspace
        viewModelScope.launch {
            val notes = withContext(Dispatchers.IO) { store.notes(workspace) }
            state = state.copy(notes = notes)
            if (state.tab == AppTab.GROUPS && state.modelReady) refreshGroups()
        }
    }

    private fun refreshDailySummaries() {
        viewModelScope.launch(Dispatchers.IO) {
            val summaries = store.dailySummaries()
            withContext(Dispatchers.Main) { state = state.copy(dailySummaries = summaries) }
        }
    }

    private fun recordPss() { peakPssKb = maxOf(peakPssKb, Debug.getPss()) }

    private fun percentile95(samples: List<Double>): Double {
        val sorted = samples.sorted()
        return sorted[ceil(sorted.size * 0.95).toInt().coerceIn(1, sorted.size) - 1]
    }

    private fun jsonNumber(value: Double): Any = if (value.isFinite()) value else JSONObject.NULL

    override fun onCleared() {
        state.importPreview?.mapNotNull { it.document }?.forEach(DocumentImportParser::cleanup)
        model.close()
        store.close()
        super.onCleared()
    }
}
