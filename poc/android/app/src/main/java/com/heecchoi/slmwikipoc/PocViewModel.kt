package com.heecchoi.slmwikipoc

import android.app.Application
import android.app.ActivityManager
import android.net.Uri
import android.os.Build
import android.os.Debug
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import kotlin.math.ceil

enum class AppTab(val label: String) { NOTES("메모"), GROUPS("묶음"), SEARCH("검색"), EVALUATION("평가") }

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
    val workspace: String = "personal",
    val clusterThreshold: Double = 0.05,
    val device: DeviceSummary? = null,
    val notes: List<Note> = emptyList(),
    val groups: List<NoteGroup> = emptyList(),
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
                state = state.copy(modelReady = true, device = deviceSummary, clusterThreshold = manifest.getDouble("cluster_threshold"), modelStatus = "학습 어댑터 checkpoint-250 · ONNX Runtime CPU")
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
        state = state.copy(workspace = workspace, hits = emptyList(), query = "", message = null)
        refreshNotes()
        if (state.tab == AppTab.GROUPS && state.modelReady) refreshGroups()
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
        if (note.workspace != "personal") return
        viewModelScope.launch(Dispatchers.IO) {
            store.delete(note.id)
            withContext(Dispatchers.Main) {
                state = state.copy(message = "메모를 삭제했습니다.")
                refreshNotes()
                if (state.tab == AppTab.GROUPS) refreshGroups()
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
        val notes = state.notes
        val threshold = state.clusterThreshold
        if (!state.modelReady) return
        viewModelScope.launch {
            val groups = withContext(Dispatchers.Default) { AverageLinkageGroups.cluster(notes, threshold) }
            state = state.copy(groups = groups)
        }
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
        model.close()
        store.close()
        super.onCleared()
    }
}
