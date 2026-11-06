package com.heecchoi.slmwikipoc

import android.app.Application
import android.net.Uri
import android.os.Environment
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import kotlin.math.roundToInt

data class ExperimentUiState(
    val items: List<ExperimentItem> = emptyList(),
    val run: ExperimentRun? = null,
    val stale: Boolean = false,
    val modelHash: String = "",
    val threshold: Double = 0.05,
    val busy: Boolean = false,
    val busyText: String? = null,
    val progress: Pair<Int, Int>? = null,
    val preview: List<ImportPreviewItem>? = null,
    val report: String? = null,
    val message: String? = null,
)

class ExperimentViewModel(application: Application) : AndroidViewModel(application) {
    private val app = application.applicationContext
    private val local = LifeServices.store(application)
    private val repo = ExperimentStore(application, local)
    private val model = LifeServices.model(application)
    private var operation: Job? = null
    private var refreshGeneration = 0
    var state by mutableStateOf(ExperimentUiState())
        private set

    init { viewModelScope.launch { refresh() } }

    private fun modelSignature(): String {
        val manifest = JSONObject(app.assets.open("model/embedding_encoder.manifest.json").bufferedReader().use { it.readText() })
        return manifest.getString("onnx_sha256") + ":" + manifest.getString("tokenizer_sha256")
    }

    private suspend fun refresh() {
        val generation = ++refreshGeneration
        try {
            val snapshot = withContext(Dispatchers.IO) {
                val hash = modelSignature()
                val items = repo.items()
                val threshold = repo.threshold()
                val run = repo.lastRun()
                Quad(items, run, threshold, hash)
            }
            if (generation == refreshGeneration) state = state.copy(items = snapshot.items, run = snapshot.run, threshold = snapshot.threshold,
                    modelHash = snapshot.hash, stale = snapshot.run?.let {
                        it.signature != ExperimentStore.signature(snapshot.items, snapshot.threshold, snapshot.hash)
                    } ?: false)
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (_: OutOfMemoryError) {
            state = state.copy(message = "메모리가 부족해 실험 목록을 불러오지 못했습니다.")
        } catch (error: Exception) {
            state = state.copy(message = "실험 목록을 불러오지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
        }
    }

    private data class Quad(val items: List<ExperimentItem>, val run: ExperimentRun?, val threshold: Double, val hash: String)

    private fun launchOperation(label: String, block: suspend () -> Unit) {
        if (operation?.isActive == true) return
        state = state.copy(busy = true, busyText = label, progress = null, message = null)
        operation = viewModelScope.launch {
            try {
                block()
            } catch (_: CancellationException) {
                state = state.copy(message = "작업을 취소했습니다. 완료된 입력과 임베딩은 유지합니다.")
            } catch (error: OutOfMemoryError) {
                state = state.copy(message = "메모리가 부족해 작업을 마치지 못했습니다. 입력을 줄인 뒤 다시 시도해 주세요.")
            } catch (error: Exception) {
                state = state.copy(message = "작업을 마치지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
            } finally {
                withContext(NonCancellable) { refresh() }
                state = state.copy(busy = false, busyText = null, progress = null)
                operation = null
            }
        }
    }

    fun clearMessage() { state = state.copy(message = null) }
    fun cancelOperation() { operation?.cancel() }
    fun permissionDenied() { state = state.copy(message = "모든 파일 접근 권한이 없어 스캔하지 않았습니다. 파일 선택은 사용할 수 있습니다.") }

    fun saveText(id: String?, kind: String, title: String, body: String, done: (Boolean) -> Unit) {
        if (state.busy) { done(false); return }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { repo.saveText(id, kind, title, body) }
                refresh()
                done(true)
            } catch (error: Exception) {
                state = state.copy(message = error.message ?: "입력을 저장하지 못했습니다.")
                done(false)
            }
        }
    }

    fun setLabel(id: String, label: String, done: (Boolean) -> Unit) {
        if (state.busy) { done(false); return }
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { repo.setLabel(id, label) }
                refresh()
                done(true)
            } catch (error: Exception) {
                state = state.copy(message = error.message ?: "정답 주제를 저장하지 못했습니다.")
                done(false)
            }
        }
    }

    fun remove(id: String) = launchOperation("항목을 삭제하는 중입니다") {
        withContext(Dispatchers.IO) { repo.remove(id) }
        state = state.copy(message = "항목을 삭제했습니다. 기기 원본과 이전 기록은 유지합니다.")
    }

    fun clear() = launchOperation("실험 목록을 비우는 중입니다") {
        withContext(Dispatchers.IO) { repo.clear() }
        state = state.copy(report = null, message = "실험 목록을 비웠습니다. 기기 원본과 이전 기록은 유지합니다.")
    }

    fun setThreshold(value: Float) {
        if (state.busy) return
        val rounded = (value.coerceIn(0f, 2f) * 100).roundToInt() / 100.0
        repo.setThreshold(rounded)
        refreshGeneration++
        state = state.copy(threshold = rounded, stale = state.run?.let {
            it.signature != ExperimentStore.signature(state.items, rounded, state.modelHash)
        } ?: false)
    }

    fun previewDocuments(uris: List<Uri>) {
        if (uris.isEmpty()) return
        launchOperation("문서를 확인하는 중입니다") {
            val previews = mutableListOf<ImportPreviewItem>()
            try {
                uris.forEachIndexed { index, uri ->
                    currentCoroutineContext().ensureActive()
                    state = state.copy(busyText = "문서 확인 ${index + 1}/${uris.size}", progress = index to uris.size)
                    val item = withContext(Dispatchers.IO) {
                        try {
                            val doc = DocumentImportParser.parse(app, uri)
                            ImportPreviewItem(doc, doc.fileName)
                        } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (error: Exception) {
                            ImportPreviewItem(null, uri.lastPathSegment?.substringAfterLast('/') ?: "선택한 파일",
                                error.message ?: "파일을 읽지 못했습니다.")
                        }
                    }
                    previews += item
                }
                state = state.copy(preview = previews)
            } catch (error: Throwable) {
                previews.mapNotNull(ImportPreviewItem::document).forEach(DocumentImportParser::cleanup)
                throw error
            }
        }
    }

    fun setImportLabel(id: String, value: String) {
        state = state.copy(preview = state.preview?.map { if (it.document?.id == id) it.copy(groundTruthGroup = value) else it })
    }

    fun cancelPreview() {
        state.preview?.mapNotNull(ImportPreviewItem::document)?.forEach(DocumentImportParser::cleanup)
        state = state.copy(preview = null)
    }

    fun confirmImport() {
        val previews = state.preview ?: return
        val documents = previews.mapNotNull { item -> item.document?.let { item to it } }
        if (documents.isEmpty()) return
        state = state.copy(preview = null)
        launchOperation("문서를 저장하는 중입니다") {
            var saved = 0
            val failures = mutableListOf<String>()
            try {
                documents.forEachIndexed { index, (item, doc) ->
                    currentCoroutineContext().ensureActive()
                    state = state.copy(busyText = "문서 저장 ${index + 1}/${documents.size}", progress = index to documents.size)
                    try {
                        withContext(Dispatchers.IO) {
                            val source = DocumentImportParser.persistSource(doc, File(app.noBackupFilesDir, "simple-experiment-documents"))
                            try { repo.addDocument(doc, source, item.groundTruthGroup) }
                            catch (error: Throwable) { source.storedPath?.let { File(it).delete() }; throw error }
                        }
                        saved++
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (error: Exception) {
                        failures += doc.fileName + ": " + (error.message ?: "저장 실패")
                    } finally { DocumentImportParser.cleanup(doc) }
                }
                state = state.copy(message = buildString {
                    append("${saved}개 문서를 추가했습니다.")
                    if (failures.isNotEmpty()) append(" 실패 ${failures.size}개: ${failures.joinToString(" · ")}")
                })
            } finally { documents.forEach { DocumentImportParser.cleanup(it.second) } }
        }
    }

    fun scanDevice() = launchOperation("공유 저장소에서 문서를 찾는 중입니다") {
        state = state.copy(report = null)
        check(Environment.isExternalStorageManager()) { "모든 파일 접근 권한이 없습니다." }
        var added = 0
        var updated = 0
        var unchanged = 0
        var removed = 0
        var pathErrors = 0
        val excluded = mutableMapOf<String, Int>()
        val discovery = DeviceDocumentScanner.discover(app)
        check(discovery.availableVolumes > 0) { "읽을 수 있는 공유 저장소가 없습니다." }
        val previous = withContext(Dispatchers.IO) { repo.scanned().associateBy(ScannedInputRecord::path) }
        val legacy = withContext(Dispatchers.IO) { local.scannedDocuments().associateBy(ScannedDocumentRecord::path) }
        val seen = HashSet<String>()
        try {
            discovery.files.forEachIndexed { index, file ->
                currentCoroutineContext().ensureActive()
                check(Environment.isExternalStorageManager(file)) { "스캔 중 저장소 접근 권한이 철회되었습니다." }
                state = state.copy(busyText = "문서 확인 ${index + 1}/${discovery.files.size}", progress = index to discovery.files.size)
                val path = try { file.canonicalPath } catch (_: Exception) {
                    pathErrors++; excluded.merge("경로 확인 실패", 1, Int::plus); return@forEachIndexed
                }
                seen += path
                val old = previous[path]
                val size = file.length()
                val modified = file.lastModified()
                if (size > DocumentImportParser.MAX_FILE_BYTES) {
                    excluded.merge("10MB 초과", 1, Int::plus)
                    if (old != null) withContext(Dispatchers.IO) { repo.markScanError(old.id, "파일 크기가 10MB를 넘습니다.") }
                    return@forEachIndexed
                }
                if (old != null && old.size == size && old.modifiedAt == modified && old.error == null) {
                    unchanged++
                    return@forEachIndexed
                }
                var parsed: ParsedDocument? = null
                try {
                    val isNew = withContext(Dispatchers.IO) {
                        val doc = DocumentImportParser.parseFile(app, file)
                        parsed = doc
                        check(file.length() == size && file.lastModified() == modified) { "읽는 동안 파일이 변경되었습니다." }
                        val source = DocumentSource(doc.fileName, doc.mimeType, pageTexts = doc.pageTexts,
                            devicePath = path, deviceSize = size, deviceModifiedAt = modified)
                        val oldLabel = legacy[path]?.let { runCatching { local.note(it.id)?.groundTruthGroup }.getOrNull() }
                        repo.saveScanned(doc, source, oldLabel)
                    }
                    if (isNew) added++ else updated++
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: Exception) {
                    val reason = (error.message ?: "읽기 실패").substringBefore(':').take(80)
                    excluded.merge(reason, 1, Int::plus)
                    if (old != null) withContext(Dispatchers.IO) { repo.markScanError(old.id, reason) }
                } finally { parsed?.let(DocumentImportParser::cleanup) }
            }
            if (discovery.complete && pathErrors == 0 && Environment.isExternalStorageManager()) {
                val context = currentCoroutineContext()
                withContext(Dispatchers.IO) {
                    removed = repo.removeMissingScanned(previous.values.filter { it.path !in seen }.map(ScannedInputRecord::id)) { context.ensureActive() }
                }
            }
            state = state.copy(report = buildString {
                append("발견 ${discovery.files.size} · 추가 $added · 갱신 $updated · 변경 없음 $unchanged · 사라진 항목 $removed")
                append("\n지원하지 않는 파일 ${discovery.unsupported}개 · 읽을 수 없는 폴더/저장소 ${discovery.unreadableDirectories}개")
                if (excluded.isNotEmpty()) append("\n제외·실패: ${excluded.entries.joinToString(" · ") { "${it.key} ${it.value}개" }}")
                if (!discovery.complete || pathErrors > 0) append("\n일부 경로에 접근하지 못해 사라진 항목은 정리하지 않았습니다.")
                append("\n군집을 계산하려면 실행 버튼을 눌러 주세요.")
            })
        } catch (cancelled: CancellationException) {
            state = state.copy(report = "스캔 취소 · 추가 $added · 갱신 $updated · 변경 없음 $unchanged. 사라진 항목은 정리하지 않았습니다.")
            throw cancelled
        }
    }

    fun runClustering() = launchOperation("모델을 확인하는 중입니다") {
        val snapshot = withContext(Dispatchers.IO) { repo.items() }
        check(snapshot.isNotEmpty()) { "먼저 실험할 글을 추가해 주세요." }
        val invalid = snapshot.filter { it.scanError != null }
        check(invalid.isEmpty()) { "읽지 못한 문서 ${invalid.size}개가 있습니다. 다시 스캔하거나 목록에서 삭제해 주세요." }
        val threshold = repo.threshold()
        withContext(Dispatchers.IO) { model.load() }
        val modelHash = withContext(Dispatchers.IO) { modelSignature() }
        check(modelHash.substringBefore(':') == model.modelHash) { "현재 모델의 해시가 일치하지 않습니다." }
        val expected = ExperimentStore.signature(snapshot, threshold, modelHash)
        val context = currentCoroutineContext()
        var embeddingMs = 0L
        var computed = 0
        var reused = 0
        val notes = mutableListOf<Note>()
        snapshot.forEachIndexed { index, item ->
            context.ensureActive()
            state = state.copy(busyText = "임베딩 ${index + 1}/${snapshot.size}", progress = index to snapshot.size)
            val cached = item.vector?.takeIf { item.modelHash == modelHash && item.pipelineVersion == ExperimentStore.PIPELINE_VERSION }
            val vector = if (cached != null) {
                reused++
                cached
            } else {
                val started = SystemClock.elapsedRealtime()
                val result = withContext(Dispatchers.IO) { model.embedCluster(item.title, item.body) { context.ensureActive() } }
                embeddingMs += SystemClock.elapsedRealtime() - started
                val saved = withContext(Dispatchers.IO) { repo.cacheVector(item, result, modelHash) }
                check(saved) { "처리 중 입력이 바뀌었습니다. 다시 실행해 주세요." }
                computed++
                result
            }
            notes += item.note(vector)
        }
        state = state.copy(busyText = "군집을 계산하는 중입니다", progress = null)
        val started = SystemClock.elapsedRealtime()
        val groups = withContext(Dispatchers.Default) { AverageLinkageGroups.clusterLarge(notes, threshold) { context.ensureActive() } }
        val clusteringMs = SystemClock.elapsedRealtime() - started
        state = state.copy(busyText = "지도와 품질 지표를 계산하는 중입니다")
        val (projection, quality) = withContext(Dispatchers.Default) {
            PcaProjector.project(notes) { context.ensureActive() } to
                ExperimentQualityCalculator.calculate(snapshot, groups) { context.ensureActive() }
        }
        context.ensureActive()
        val saved = withContext(Dispatchers.IO) {
            repo.saveRun(expected, modelHash, threshold, groups, projection, quality, embeddingMs, clusteringMs, computed, reused) { context.ensureActive() }
        }
        check(saved) { "처리 중 입력 또는 기준이 바뀌었습니다. 다시 실행해 주세요." }
        state = state.copy(message = "군집 실행을 마쳤습니다.")
    }

    override fun onCleared() {
        state.preview?.mapNotNull(ImportPreviewItem::document)?.forEach(DocumentImportParser::cleanup)
        super.onCleared()
    }
}
