package com.heecchoi.slmwikipoc

import android.app.Application
import android.app.NotificationManager
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class LifeAppChoice(val packageName: String, val label: String, val selected: Boolean)
data class LifeDisplayGroup(val id: String?, val title: String, val recordIds: List<String>, val automatic: Boolean)

data class LifeUiState(
    val records: List<LifeRecord> = emptyList(),
    val candidates: List<LifeCandidate> = emptyList(),
    val tasks: List<LifeTask> = emptyList(),
    val topics: List<LifeDisplayGroup> = emptyList(),
    val orders: List<LifeDisplayGroup> = emptyList(),
    val modelGroups: List<List<String>> = emptyList(),
    val hybridGroups: List<List<String>> = emptyList(),
    val realModelMetrics: LifePairMetrics? = null,
    val realHybridMetrics: LifePairMetrics? = null,
    val searchResults: List<LifeRecord> = emptyList(),
    val documents: List<Note> = emptyList(),
    val apps: List<LifeAppChoice> = emptyList(),
    val allowed: Set<String> = emptySet(),
    val permissionGranted: Boolean = false,
    val connected: Boolean = false,
    val collecting: Boolean = false,
    val excluded: Int = 0,
    val threshold: Double? = null,
    val topicThreshold: Double? = null,
    val computing: Boolean = false,
    val searchActive: Boolean = false,
    val searchPerformed: Boolean = false,
    val message: String? = null,
    val usage: List<String> = emptyList(),
    val devReviewed: Set<String> = emptySet(),
    val testReviewed: Set<String> = emptySet(),
    val evaluation: LifeEvaluationResult? = null,
    val modelHash: String? = null,
    val evaluationProgress: String? = null,
    val trialStart: Long? = null,
    val versions: List<LifeVersion> = emptyList(),
)

class LifeViewModel(application: Application) : AndroidViewModel(application) {
    private val repo = LifeServices.repository(application)
    private val store = LifeServices.store(application)
    private val model = LifeServices.model(application)
    private var refreshJob: Job? = null
    private var evaluationJob: Job? = null
    private var expiryJob: Job? = null
    var state by mutableStateOf(LifeUiState())
        private set

    init {
        LifeWork.scheduleCleanup(application)
        viewModelScope.launch { LifeSignals.revision.collectLatest { refresh() } }
        viewModelScope.launch {
            LifeSignals.connected.collectLatest { connected -> state = state.copy(connected = connected) }
        }
    }

    fun onResume() {
        state = state.copy(records = emptyList(), candidates = emptyList(), tasks = emptyList(), topics = emptyList(),
            orders = emptyList(), modelGroups = emptyList(), hybridGroups = emptyList(), searchResults = emptyList(),
            versions = emptyList(), computing = true)
        refresh()
        loadApps()
    }

    fun refresh() {
        refreshJob?.cancel()
        refreshJob = viewModelScope.launch {
            try {
                val snapshot = withContext(Dispatchers.IO) {
                    Snapshot(repo.records(), repo.candidates(), repo.tasks(), repo.manualGroups("topic"),
                        repo.manualGroups("order"), repo.allowedPackages(), repo.collectionEnabled(),
                        repo.threshold(), repo.topicThreshold(), repo.usageSummary(),
                        repo.reviewedIds(LifeSynthetic.DEV), repo.reviewedIds(LifeSynthetic.TEST),
                        repo.lastEvaluation(), repo.currentModelHash(), repo.trialStart(), repo.nextExpiry())
                }
                val byId = snapshot.records.associateBy(LifeRecord::id)
                val granted = getApplication<Application>().getSystemService(NotificationManager::class.java)
                    .isNotificationListenerAccessGranted(ComponentName(getApplication(), LifeNotificationListener::class.java))
                state = state.copy(records = snapshot.records, candidates = snapshot.candidates, tasks = snapshot.tasks,
                    allowed = snapshot.allowed, collecting = snapshot.collecting, permissionGranted = granted,
                    connected = LifeSignals.connected.value, excluded = repo.excludedCount(),
                    threshold = snapshot.threshold, topicThreshold = snapshot.topicThreshold,
                    computing = true, usage = snapshot.usage, devReviewed = snapshot.devReviewed,
                    testReviewed = snapshot.testReviewed, evaluation = snapshot.evaluation, modelHash = snapshot.modelHash, trialStart = snapshot.trialStart,
                    searchResults = state.searchResults.mapNotNull { byId[it.id] })
                scheduleExpiry(snapshot.nextExpiry)
                val ready = snapshot.records.all { it.clusterVector != null && it.processing == "ready" }
                if (!ready) {
                    state = state.copy(computing = false, topics = overlay(emptyList(), snapshot.topics, snapshot.records),
                        orders = overlay(emptyList(), snapshot.orders, snapshot.records),
                        modelGroups = emptyList(), hybridGroups = emptyList(), realModelMetrics = null, realHybridMetrics = null)
                    return@launch
                }
                val result = withContext(Dispatchers.Default) {
                    val context = currentCoroutineContext()
                    val threshold = snapshot.threshold ?: 0.05
                    val topicThreshold = snapshot.topicThreshold ?: threshold
                    val modelGroups = LifeGrouping.model(snapshot.records, threshold) { context.ensureActive() }
                    val topicGroups = if (topicThreshold == threshold) modelGroups
                        else LifeGrouping.model(snapshot.records, topicThreshold) { context.ensureActive() }
                    val hybrid = LifeGrouping.hybrid(snapshot.records, modelGroups, threshold)
                    GroupResult(modelGroups, topicGroups, hybrid, LifeGrouping.metrics(snapshot.records, modelGroups),
                        LifeGrouping.metrics(snapshot.records, hybrid))
                }
                state = state.copy(computing = false, modelGroups = result.model, hybridGroups = result.hybrid,
                    realModelMetrics = result.modelMetrics, realHybridMetrics = result.hybridMetrics,
                    topics = overlay(result.topics, snapshot.topics, snapshot.records),
                    orders = overlay(result.hybrid, snapshot.orders, snapshot.records))
            } catch (_: CancellationException) {
                // 새 변경이 이전 계산을 대체한다.
            } catch (error: Throwable) {
                state = state.copy(computing = false, topics = emptyList(), orders = emptyList(),
                    modelGroups = emptyList(), hybridGroups = emptyList(), realModelMetrics = null, realHybridMetrics = null,
                    message = "생활 기록을 불러오거나 군집화하지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
            }
        }
    }

    private data class Snapshot(
        val records: List<LifeRecord>, val candidates: List<LifeCandidate>, val tasks: List<LifeTask>,
        val topics: List<LifeManualGroup>, val orders: List<LifeManualGroup>, val allowed: Set<String>,
        val collecting: Boolean, val threshold: Double?, val topicThreshold: Double?, val usage: List<String>,
        val devReviewed: Set<String>, val testReviewed: Set<String>, val evaluation: LifeEvaluationResult?, val modelHash: String, val trialStart: Long?,
        val nextExpiry: Long?,
    )

    private data class GroupResult(
        val model: List<List<String>>, val topics: List<List<String>>, val hybrid: List<List<String>>,
        val modelMetrics: LifePairMetrics, val hybridMetrics: LifePairMetrics,
    )

    private fun scheduleExpiry(expiry: Long?) {
        expiryJob?.cancel()
        if (expiry == null) return
        expiryJob = viewModelScope.launch {
            delay((expiry - System.currentTimeMillis()).coerceAtLeast(1L))
            withContext(Dispatchers.IO) { repo.cleanupExpired() }
            refresh()
        }
    }

    private fun overlay(raw: List<List<String>>, manual: List<LifeManualGroup>, records: List<LifeRecord>): List<LifeDisplayGroup> {
        val byId = records.associateBy(LifeRecord::id)
        val primary = manual.flatMap { group -> group.members.filter { it.second == "primary" }.map { it.first } }.toSet()
        val corrected = manual.map { group ->
            LifeDisplayGroup(group.id, group.title, group.members.map { it.first }.filter { it in byId }, false)
        }
        val automatic = raw.mapNotNull { group ->
            val ids = group.filter { it in byId && it !in primary }
            if (ids.isEmpty()) null else LifeDisplayGroup(null, byId.getValue(ids.first()).title.ifBlank { "이름 미정" }, ids, true)
        }
        return (corrected + automatic).sortedWith(compareByDescending<LifeDisplayGroup> { it.recordIds.size }.thenBy { it.title })
    }

    private fun action(block: () -> Unit) {
        viewModelScope.launch {
            try { withContext(Dispatchers.IO) { block() }; refresh() }
            catch (error: Exception) { state = state.copy(message = error.message ?: "작업을 완료하지 못했습니다.") }
        }
    }

    fun clearMessage() { state = state.copy(message = null) }
    fun saveText(id: String?, title: String, body: String, shared: Boolean, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    if (id == null) repo.addText(title, body, shared) else repo.updateText(id, title, body)
                }
                onComplete(true)
                refresh()
            } catch (error: Exception) {
                state = state.copy(message = error.message ?: "기록을 저장하지 못했습니다.")
                onComplete(false)
            }
        }
    }
    fun linkDocument(id: String) = action { repo.linkDocument(id) }
    fun deleteRecord(id: String) = action { repo.deleteRecord(id) }
    fun deleteNotifications() = action { repo.deleteNotifications() }
    fun retryFailures() = action { repo.retryFailures() }
    fun approve(candidate: LifeCandidate, kind: String, title: String, date: String?, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            try { withContext(Dispatchers.IO) { repo.approveCandidate(candidate, kind, title, date) }; onComplete(true); refresh() }
            catch (error: Exception) { state = state.copy(message = error.message ?: "후보를 저장하지 못했습니다."); onComplete(false) }
        }
    }
    fun dismissCandidate(id: Long) = action { repo.dismissCandidate(id) }
    fun saveTask(id: String?, kind: String, title: String, date: String?, onComplete: (Boolean) -> Unit) {
        viewModelScope.launch {
            try { withContext(Dispatchers.IO) { repo.saveTask(id, kind, title, date) }; onComplete(true); refresh() }
            catch (error: Exception) { state = state.copy(message = error.message ?: "일정을 저장하지 못했습니다."); onComplete(false) }
        }
    }
    fun setDone(id: String, done: Boolean) = action { repo.setTaskDone(id, done) }
    fun deleteTask(id: String) = action { repo.deleteTask(id) }
    fun setTruth(id: String, value: String) = action { repo.setTruthOrder(id, value) }
    fun logUsage(scenario: String, method: String, success: Boolean, seconds: Int?, issue: String) =
        action { repo.saveUsage(scenario, method, success, seconds, issue) }
    fun startTrial() = action { repo.startTrial() }
    fun setReviewed(dataset: String, orderId: String, reviewed: Boolean) = action { repo.setReviewed(dataset, orderId, reviewed) }

    fun loadDocuments() {
        viewModelScope.launch {
            val notes = withContext(Dispatchers.IO) { store.notes("experiment") }
            state = state.copy(documents = notes)
        }
    }

    fun loadVersions(recordId: String) {
        viewModelScope.launch {
            val versions = withContext(Dispatchers.IO) { repo.versions(recordId) }
            state = state.copy(versions = versions)
        }
    }

    fun createExtraTopic(title: String, recordId: String) = action {
        val id = repo.createGroup("topic", title, emptyList())
        repo.addMember(id, recordId, extra = true)
    }

    fun createGroup(kind: String, title: String, records: List<String>) = action { repo.createGroup(kind, title, records) }

    fun renameGroup(kind: String, group: LifeDisplayGroup, title: String) = action {
        val id = group.id ?: repo.createGroup(kind, if (kind == "topic") "사용자 주제" else "사용자 주문", group.recordIds)
        repo.renameGroup(id, title)
    }

    fun addToGroup(kind: String, group: LifeDisplayGroup, recordId: String, extra: Boolean = false) = action {
        val id = group.id ?: repo.createGroup(kind, if (kind == "topic") "사용자 주제" else "사용자 주문", group.recordIds)
        repo.addMember(id, recordId, extra)
    }
    fun removeFromGroup(groupId: String, recordId: String) = action { repo.removeMember(groupId, recordId) }

    fun splitRecord(kind: String, record: LifeRecord) = action {
        repo.createGroup(kind, if (kind == "topic") "분리한 주제" else "분리한 주문", listOf(record.id))
    }

    fun relatedGroups(record: LifeRecord, kind: String): List<LifeDisplayGroup> {
        val groups = if (kind == "topic") state.topics else state.orders
        val threshold = if (kind == "topic") state.topicThreshold ?: 0.05 else state.threshold ?: 0.05
        val candidates = LifeGrouping.candidates(record, groups.map { it.recordIds }, state.records, threshold, kind == "order")
        return groups.filter { it.recordIds in candidates }
    }

    fun search(query: String) {
        val clean = query.trim()
        if (clean.isBlank()) { state = state.copy(searchResults = emptyList(), searchPerformed = false); return }
        viewModelScope.launch {
            state = state.copy(searchActive = true)
            try {
                val records = state.records
                val found = withContext(Dispatchers.IO) {
                    val ready = records.filter { it.clusterVector != null && it.searchVectors.isNotEmpty() }
                    val ranked = if (ready.isEmpty()) emptyList() else {
                        model.load()
                        val queryVector = model.embedQuery(clean)
                        val notes = ready.map { record -> Note(record.id, "life", null, record.title, record.text,
                            record.createdAt, record.updatedAt, record.searchVectors, requireNotNull(record.clusterVector)) }
                        val byId = records.associateBy(LifeRecord::id)
                        NoteSearch.search(notes, clean, queryVector, SearchMode.HYBRID, limit = 40).mapNotNull { byId[it.note.id] }
                    }
                    val pendingMatches = records.filter { it.clusterVector == null && (it.title.contains(clean, true) || it.text.contains(clean, true)) }
                    (ranked + pendingMatches).distinctBy(LifeRecord::id)
                }
                val current = state.records.associateBy(LifeRecord::id)
                state = state.copy(searchActive = false, searchPerformed = true, searchResults = found.mapNotNull { current[it.id] })
            } catch (error: Exception) {
                state = state.copy(searchActive = false, message = "검색하지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
            }
        }
    }

    fun saveComparison() = action {
        val current = state
        check(!current.computing && current.records.all { it.clusterVector != null }) { "기록 처리가 끝난 뒤 비교해 주세요." }
        val threshold = current.threshold ?: error("먼저 개발 자료로 기준값을 정해 주세요.")
        model.load()
        val dataHash = LifeRepository.digest(current.records.sortedBy(LifeRecord::id).joinToString("|") {
            it.id + ":" + it.updatedAt + ":" + LifeRepository.digest(it.title + "\u0000" + it.text)
        })
        repo.saveRun(threshold, dataHash, model.modelHash, current.modelGroups, current.hybridGroups)
    }

    fun loadApps() {
        viewModelScope.launch {
            val app = getApplication<Application>()
            val choices = withContext(Dispatchers.IO) {
                val pm = app.packageManager
                val intent = Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
                @Suppress("DEPRECATION")
                val packages = pm.queryIntentActivities(intent, 0).map { it.activityInfo.packageName }.toSet() +
                    LifeNotificationListener.activePackages()
                val allowed = repo.allowedPackages()
                packages.filter { it != app.packageName }.map { pkg ->
                    val label = runCatching {
                        @Suppress("DEPRECATION")
                        pm.getApplicationLabel(pm.getApplicationInfo(pkg, 0)).toString()
                    }.getOrDefault(pkg)
                    LifeAppChoice(pkg, label, pkg in allowed)
                }.sortedBy { it.label.lowercase() }
            }
            state = state.copy(apps = choices)
        }
    }

    fun setAllowed(packageName: String, allowed: Boolean) {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { repo.setAllowed(packageName, allowed) }
            if (repo.collectionEnabled()) LifeNotificationListener.captureCurrent(getApplication())
            loadApps()
            refresh()
        }
    }

    fun setCollecting(enabled: Boolean) {
        val app = getApplication<Application>()
        val granted = app.getSystemService(NotificationManager::class.java)
            .isNotificationListenerAccessGranted(ComponentName(app, LifeNotificationListener::class.java))
        if (enabled && !granted) { state = state.copy(message = "Android 설정에서 알림 접근을 허용해 주세요."); return }
        if (enabled && repo.allowedPackages().isEmpty()) { state = state.copy(message = "수집할 앱을 하나 이상 선택해 주세요."); return }
        repo.setCollectionEnabled(enabled)
        if (enabled) LifeNotificationListener.captureCurrent(app)
        refresh()
    }

    fun cancelEvaluation() { evaluationJob?.cancel() }

    private suspend fun syntheticRecords(dataset: String): List<LifeRecord> {
        val items = LifeSynthetic.items(dataset)
        withContext(Dispatchers.IO) { model.load() }
        return items.mapIndexed { index, item ->
            currentCoroutineContext().ensureActive()
            withContext(Dispatchers.Main) {
                state = state.copy(evaluationProgress = "가상 자료 임베딩 " + (index + 1) + "/" + items.size)
            }
            val vectors = withContext(Dispatchers.IO) { model.embedNote(item.title, item.text) }
            LifeRecord(item.id, "synthetic", item.title, item.text, 0L, 0L, item.merchant, null, null,
                LifeRules.orderKey(item.title, item.text, item.merchant), vectors.searchChunks, vectors.cluster, "ready", null, item.orderId)
        }
    }

    fun calibrate() {
        if (evaluationJob?.isActive == true) return
        if (!repo.reviewedIds(LifeSynthetic.DEV).containsAll(LifeSynthetic.orders(LifeSynthetic.DEV).keys)) {
            state = state.copy(message = "가상 개발 자료의 주문을 모두 확인해 주세요.")
            return
        }
        if (repo.trialActive()) { state = state.copy(message = "7일 시험 중에는 기준값을 변경할 수 없습니다."); return }
        state = state.copy(evaluationProgress = "개발 자료를 준비하는 중입니다")
        evaluationJob = viewModelScope.launch {
            try {
                val records = syntheticRecords(LifeSynthetic.DEV)
                val topicLabels = LifeSynthetic.items(LifeSynthetic.DEV).associate { it.id to it.topic }
                val topicRecords = records.map { it.copy(truthOrder = topicLabels.getValue(it.id)) }
                val thresholds = withContext(Dispatchers.Default) {
                    val context = currentCoroutineContext()
                    var bestOrder = 0.0
                    var bestTopic = 0.0
                    var orderScore = -1.0
                    var topicScore = -1.0
                    for (step in 0..200) {
                        context.ensureActive()
                        val threshold = step / 100.0
                        val groups = LifeGrouping.model(records, threshold) { context.ensureActive() }
                        val order = LifeGrouping.metrics(records, groups).f1 ?: 0.0
                        val topic = LifeGrouping.metrics(topicRecords, groups).f1 ?: 0.0
                        if (order > orderScore) { orderScore = order; bestOrder = threshold }
                        if (topic > topicScore) { topicScore = topic; bestTopic = threshold }
                        if (step % 20 == 0) withContext(Dispatchers.Main) {
                            state = state.copy(evaluationProgress = "개발 자료 기준값 비교 " + step + "/200")
                        }
                    }
                    bestOrder to bestTopic
                }
                withContext(Dispatchers.IO) { repo.setThreshold(thresholds.first, thresholds.second, model.modelHash) }
                state = state.copy(evaluationProgress = null, message = "개발 자료의 기준값을 정했습니다. 평가 자료를 확인해 주세요.")
                refresh()
            } catch (_: CancellationException) {
                state = state.copy(evaluationProgress = null, message = "기준값 계산을 중단했습니다.")
            } catch (error: Throwable) {
                state = state.copy(evaluationProgress = null, message = "기준값을 정하지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
            }
        }
    }

    fun evaluateSynthetic() {
        if (evaluationJob?.isActive == true) return
        val threshold = repo.threshold()
        if (threshold == null) { state = state.copy(message = "먼저 개발 자료로 기준값을 정해 주세요."); return }
        if (!repo.reviewedIds(LifeSynthetic.TEST).containsAll(LifeSynthetic.orders(LifeSynthetic.TEST).keys)) {
            state = state.copy(message = "가상 평가 자료의 주문을 모두 확인해 주세요.")
            return
        }
        state = state.copy(evaluationProgress = "평가 자료를 준비하는 중입니다")
        evaluationJob = viewModelScope.launch {
            try {
                val records = syntheticRecords(LifeSynthetic.TEST)
                val metrics = withContext(Dispatchers.Default) {
                    val context = currentCoroutineContext()
                    val modelGroups = LifeGrouping.model(records, threshold) { context.ensureActive() }
                    val hybridGroups = LifeGrouping.hybrid(records, modelGroups, threshold)
                    LifeGrouping.metrics(records, modelGroups) to LifeGrouping.metrics(records, hybridGroups)
                }
                withContext(Dispatchers.IO) { repo.saveEvaluation(LifeSynthetic.TEST, threshold, model.modelHash, metrics.first, metrics.second) }
                state = state.copy(evaluationProgress = null, message = "가상 평가를 마쳤습니다. 실제 7일 결과와 따로 확인하세요.")
                refresh()
            } catch (_: CancellationException) {
                state = state.copy(evaluationProgress = null, message = "평가를 중단했습니다.")
            } catch (error: Throwable) {
                state = state.copy(evaluationProgress = null, message = "평가하지 못했습니다: " + (error.message ?: error.javaClass.simpleName))
            }
        }
    }
}
