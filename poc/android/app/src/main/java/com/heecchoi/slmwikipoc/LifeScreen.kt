package com.heecchoi.slmwikipoc

import android.content.Intent
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.time.LocalDate
import java.util.Date
import java.util.Locale

data class SharedLifeText(val title: String, val body: String)

private fun groupKey(group: LifeDisplayGroup): String =
    group.id ?: "auto:" + LifeRepository.digest(group.recordIds.sorted().joinToString("|"))

private fun lifeDate(value: Long): String = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.KOREA).format(Date(value))

@Composable
fun LifeScreen(viewModel: LifeViewModel, shared: SharedLifeText?, onSharedDone: () -> Unit, onOpenDocuments: () -> Unit) {
    val state = viewModel.state
    var page by rememberSaveable { mutableStateOf("home") }
    var previousPage by rememberSaveable { mutableStateOf("home") }
    var topicKey by rememberSaveable { mutableStateOf<String?>(null) }
    var orderKey by rememberSaveable { mutableStateOf<String?>(null) }
    var selectedRecordId by rememberSaveable { mutableStateOf<String?>(null) }
    var showNewRecord by rememberSaveable { mutableStateOf(false) }
    var editingRecord by remember { mutableStateOf<LifeRecord?>(null) }
    var selectedCandidate by remember { mutableStateOf<LifeCandidate?>(null) }
    var editingTask by remember { mutableStateOf<LifeTask?>(null) }
    var showNewTask by rememberSaveable { mutableStateOf(false) }
    var choiceKind by rememberSaveable { mutableStateOf<String?>(null) }
    var renameGroup by remember { mutableStateOf<Pair<String, LifeDisplayGroup>?>(null) }
    var deleteRecord by remember { mutableStateOf<LifeRecord?>(null) }
    var clearNotifications by rememberSaveable { mutableStateOf(false) }
    val topic = state.topics.firstOrNull { groupKey(it) == topicKey }
    val order = state.orders.firstOrNull { groupKey(it) == orderKey }
    val selectedRecord = state.records.firstOrNull { it.id == selectedRecordId }
    val context = LocalContext.current
    val permissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { viewModel.onResume() }

    fun openRecord(record: LifeRecord, from: String) {
        selectedRecordId = record.id
        previousPage = from
        page = "record"
        if (record.kind == "notification") viewModel.loadVersions(record.id)
    }
    fun back() {
        page = when (page) {
            "record" -> previousPage
            "order" -> if (topicKey != null) "topic" else "home"
            "topic" -> "topics"
            "dev-review", "test-review" -> "evaluation"
            else -> "home"
        }
    }
    BackHandler(enabled = page != "home") { back() }
    LaunchedEffect(Unit) { viewModel.onResume() }
    LaunchedEffect(shared) { if (shared != null) page = "home" }
    LaunchedEffect(page, selectedRecord?.id, selectedRecord?.updatedAt) {
        if (page == "record" && selectedRecord?.kind == "notification") viewModel.loadVersions(selectedRecord.id)
    }

    when (page) {
        "home" -> LifeHomePage(state, viewModel, onPage = { page = it }, onRecord = { openRecord(it, "home") },
            onOrder = { topicKey = null; orderKey = groupKey(it); page = "order" })
        "records" -> LifeRecordsPage(state, onBack = { page = "home" }, onNew = { showNewRecord = true },
            onDocuments = { viewModel.loadDocuments(); page = "documents" }, onRecord = { openRecord(it, "records") })
        "documents" -> LifeDocumentsPage(state, onBack = { page = "records" }, onLink = viewModel::linkDocument, onOpenDocuments = onOpenDocuments)
        "topics" -> LifeTopicsPage(state, onBack = { page = "home" }, onTopic = { topicKey = groupKey(it); page = "topic" })
        "topic" -> LifeTopicPage(state, topic, onBack = { page = "topics" },
            onRename = { renameGroup = "topic" to it },
            onOrder = { orderKey = groupKey(it); page = "order" }, onRecord = { openRecord(it, "topic") })
        "order" -> LifeOrderPage(state, order, topic, onBack = { back() }, onRename = { renameGroup = "order" to it },
            onRecord = { openRecord(it, "order") })
        "record" -> LifeRecordPage(state, selectedRecord, onBack = { back() }, onEdit = { editingRecord = it },
            onDelete = { deleteRecord = it }, onChoose = { choiceKind = it },
            onSplit = { kind -> selectedRecord?.let { viewModel.splitRecord(kind, it) } },
            onTruth = { value -> selectedRecord?.let { viewModel.setTruth(it.id, value) } },
            onRemoveTopic = { groupId -> selectedRecord?.let { viewModel.removeFromGroup(groupId, it.id) } })
        "tasks" -> LifeTasksPage(state, onBack = { page = "home" }, onNew = { showNewTask = true },
            onCandidate = { selectedCandidate = it }, onDismiss = viewModel::dismissCandidate,
            onEdit = { editingTask = it }, onDone = viewModel::setDone, onDelete = viewModel::deleteTask,
            onSource = { id -> state.records.firstOrNull { it.id == id }?.let { openRecord(it, "tasks") } })
        "settings" -> LifeSettingsPage(state, onBack = { page = "home" },
            onPermission = { permissionLauncher.launch(Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS)) },
            onAllowed = viewModel::setAllowed, onCollect = viewModel::setCollecting, onReload = viewModel::onResume,
            onClear = { clearNotifications = true }, onRetry = viewModel::retryFailures)
        "evaluation" -> LifeEvaluationPage(state, viewModel, onBack = { page = "home" }, onReview = { page = it },
            onRecord = { openRecord(it, "evaluation") })
        "dev-review", "test-review" -> LifeReviewPage(
            if (page == "dev-review") LifeSynthetic.DEV else LifeSynthetic.TEST, state, viewModel, onBack = { page = "evaluation" },
        )
        "trial" -> LifeTrialPage(state, viewModel, onBack = { page = "home" })
    }

    state.message?.let { message ->
        AlertDialog(onDismissRequest = viewModel::clearMessage, title = { Text("생활 기록") }, text = { Text(message) },
            confirmButton = { TextButton(onClick = viewModel::clearMessage) { Text("확인") } })
    }
    if (showNewRecord) LifeTextEditor(null, null, onDismiss = { showNewRecord = false }, onSave = { title, body, done ->
        viewModel.saveText(null, title, body, false) { ok -> if (ok) showNewRecord = false; done(ok) }
    })
    editingRecord?.let { record ->
        LifeTextEditor(record, null, onDismiss = { editingRecord = null }, onSave = { title, body, done ->
            viewModel.saveText(record.id, title, body, record.kind == "share") { ok -> if (ok) editingRecord = null; done(ok) }
        })
    }
    if (shared != null) LifeTextEditor(null, shared, onDismiss = onSharedDone, onSave = { title, body, done ->
        viewModel.saveText(null, title, body, true) { ok -> if (ok) onSharedDone(); done(ok) }
    })
    selectedCandidate?.let { candidate ->
        LifeTaskEditor(candidate.kind, candidate.title, candidate.dueDate, onDismiss = { selectedCandidate = null },
            onSave = { kind, title, date, done -> viewModel.approve(candidate, kind, title, date) { if (it) selectedCandidate = null; done(it) } })
    }
    editingTask?.let { task ->
        LifeTaskEditor(task.kind, task.title, task.dueDate, onDismiss = { editingTask = null },
            onSave = { kind, title, date, done -> viewModel.saveTask(task.id, kind, title, date) { if (it) editingTask = null; done(it) } })
    }
    if (showNewTask) LifeTaskEditor("task", "", null, onDismiss = { showNewTask = false },
        onSave = { kind, title, date, done -> viewModel.saveTask(null, kind, title, date) { if (it) showNewTask = false; done(it) } })
    choiceKind?.let { choice ->
        selectedRecord?.let { record ->
            val kind = if (choice == "order") "order" else "topic"
            val groups = if (kind == "topic") state.topics else state.orders
            LifeGroupChoiceDialog(record, kind, choice == "extra", groups, viewModel.relatedGroups(record, kind),
                onDismiss = { choiceKind = null },
                onChoose = { group -> viewModel.addToGroup(kind, group, record.id, extra = choice == "extra"); choiceKind = null },
                onCreate = { name ->
                    if (choice == "extra") viewModel.createExtraTopic(name, record.id)
                    else viewModel.createGroup(kind, name, listOf(record.id))
                    choiceKind = null
                })
        }
    }
    renameGroup?.let { (kind, group) ->
        LifeNameDialog(group.title, onDismiss = { renameGroup = null }, onSave = { title ->
            viewModel.renameGroup(kind, group, title); renameGroup = null; page = "topics"
        })
    }
    deleteRecord?.let { record ->
        AlertDialog(onDismissRequest = { deleteRecord = null }, title = { Text("생활 기록을 삭제할까요?") },
            text = { Text(if (record.kind == "document") "생활 화면의 연결만 삭제합니다. 기존 문서와 승인한 할 일은 유지합니다."
                else "앱에 저장한 기록과 미확인 후보를 삭제합니다. 원래 앱의 자료와 승인한 할 일은 유지합니다.") },
            confirmButton = { TextButton(onClick = { viewModel.deleteRecord(record.id); deleteRecord = null; page = "records" }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleteRecord = null }) { Text("취소") } })
    }
    if (clearNotifications) AlertDialog(onDismissRequest = { clearNotifications = false },
        title = { Text("수집한 알림을 모두 삭제할까요?") },
        text = { Text("생활 POC에 저장한 알림과 미확인 후보를 지웁니다. 승인한 할 일과 원래 앱의 알림은 유지합니다.") },
        confirmButton = { TextButton(onClick = { viewModel.deleteNotifications(); clearNotifications = false }) { Text("삭제") } },
        dismissButton = { TextButton(onClick = { clearNotifications = false }) { Text("취소") } })
}

@Composable
private fun LifeHomePage(
    state: LifeUiState, viewModel: LifeViewModel, onPage: (String) -> Unit,
    onRecord: (LifeRecord) -> Unit, onOrder: (LifeDisplayGroup) -> Unit,
) {
    var query by rememberSaveable { mutableStateOf("") }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text("오늘의 기록", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold) }
        item { Text("메모와 선택한 문서·알림을 기기 안에서 정리합니다.", style = MaterialTheme.typography.bodySmall) }
        if (state.computing || state.records.any { it.processing == "pending" }) item {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text("기록을 처리하는 중입니다. 완료 전 군집은 전체 결과가 아닙니다.", style = MaterialTheme.typography.bodySmall)
        }
        if (state.records.any { it.processing == "failed" }) item {
            Text("처리하지 못한 기록이 있습니다. 알림 설정에서 다시 시도할 수 있습니다.", color = MaterialTheme.colorScheme.error)
        }
        item {
            OutlinedTextField(query, { query = it }, Modifier.fillMaxWidth(), label = { Text("생활 기록 검색") }, singleLine = true)
            Button(onClick = { viewModel.search(query) }, enabled = !state.searchActive) { Text("검색") }
        }
        if (state.searchResults.isNotEmpty()) {
            item { Text("검색 후보 · 원문을 확인하세요", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
            items(state.searchResults, key = { "search-" + it.id }) { record -> LifeRecordCard(record) { onRecord(record) } }
        }
        if (state.searchPerformed && state.searchResults.isEmpty() && !state.searchActive) item { Text("검색 결과가 없습니다.") }
        item { Text("최근 주문·연결 후보", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        if (state.orders.isEmpty()) item { Text("아직 표시할 주문이 없습니다.", style = MaterialTheme.typography.bodySmall) }
        items(state.orders.take(5), key = { "order-" + groupKey(it) }) { group ->
            LifeGroupCard(group, if (group.automatic && group.recordIds.size == 1) "연결 확인 필요" else "주문") { onOrder(group) }
        }
        item { Text("예정된 일정과 할 일", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        items(state.tasks.filterNot(LifeTask::done).take(5), key = { "task-" + it.id }) { task ->
            ElevatedCard(Modifier.fillMaxWidth().clickable { onPage("tasks") }) {
                Text(task.title + " · " + (task.dueDate ?: "날짜 미정"), Modifier.padding(14.dp))
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(onClick = { onPage("records") }, modifier = Modifier.fillMaxWidth()) { Text("생활 기록") }
                OutlinedButton(onClick = { onPage("topics") }, modifier = Modifier.fillMaxWidth()) { Text("주제와 주문") }
                OutlinedButton(onClick = { onPage("tasks") }, modifier = Modifier.fillMaxWidth()) { Text("일정·할 일과 후보 (" + state.candidates.size + ")") }
                OutlinedButton(onClick = { onPage("settings") }, modifier = Modifier.fillMaxWidth()) { Text("알림 수집 설정") }
                OutlinedButton(onClick = { onPage("evaluation") }, modifier = Modifier.fillMaxWidth()) { Text("생활 군집 비교") }
                OutlinedButton(onClick = { onPage("trial") }, modifier = Modifier.fillMaxWidth()) { Text("7일 사용 기록") }
            }
        }
    }
}

@Composable
private fun LifeRecordCard(record: LifeRecord, onClick: () -> Unit) {
    ElevatedCard(Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(record.title, fontWeight = FontWeight.SemiBold)
            Text((record.sourceName ?: when (record.kind) {
                "memo" -> "직접 작성"; "share" -> "공유한 글"; "document" -> "선택한 문서"; else -> "알림"
            }) + if (record.processing == "ready") "" else " · " + if (record.processing == "failed") "처리 실패" else "처리 대기",
                style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun LifeGroupCard(group: LifeDisplayGroup, label: String, onClick: () -> Unit) {
    ElevatedCard(Modifier.fillMaxWidth().clickable(onClick = onClick)) {
        Column(Modifier.padding(14.dp)) {
            Text(group.title, fontWeight = FontWeight.SemiBold)
            Text(label + " · " + group.recordIds.size + "개 기록", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun LifeRecordsPage(state: LifeUiState, onBack: () -> Unit, onNew: () -> Unit, onDocuments: () -> Unit, onRecord: (LifeRecord) -> Unit) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("생활 기록 " + state.records.size + "개", style = MaterialTheme.typography.headlineSmall) }
        item { Button(onClick = onNew) { Text("메모 작성") } }
        item { OutlinedButton(onClick = onDocuments) { Text("기존 문서 연결") } }
        if (state.records.isEmpty()) item { Text("메모를 작성하거나 문서·알림을 추가해 보세요.") }
        items(state.records, key = LifeRecord::id) { record -> LifeRecordCard(record) { onRecord(record) } }
    }
}

@Composable
private fun LifeDocumentsPage(state: LifeUiState, onBack: () -> Unit, onLink: (String) -> Unit, onOpenDocuments: () -> Unit) {
    val linked = state.records.mapNotNull(LifeRecord::noteId).toSet()
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 기록") } }
        item { Text("기존 문서 연결", style = MaterialTheme.typography.headlineSmall) }
        item { Text("선택한 문서는 원래 앱 내 기록을 참조합니다. 기존 문서가 삭제되면 연결도 사라집니다.", style = MaterialTheme.typography.bodySmall) }
        item { OutlinedButton(onClick = onOpenDocuments) { Text("문서 가져오기·스캔 화면") } }
        if (state.documents.isEmpty()) item { Text("연결할 문서가 없습니다.") }
        items(state.documents, key = Note::id) { note ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                    Text(note.title, Modifier.weight(1f))
                    TextButton(onClick = { onLink(note.id) }, enabled = note.id !in linked) { Text(if (note.id in linked) "연결됨" else "연결") }
                }
            }
        }
    }
}

@Composable
private fun LifeTopicsPage(state: LifeUiState, onBack: () -> Unit, onTopic: (LifeDisplayGroup) -> Unit) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("주제와 주문", style = MaterialTheme.typography.headlineSmall) }
        if (state.threshold == null) item { Text("아직 기준을 확인하기 전인 임시 묶음입니다. 군집 비교에서 개발 자료를 검토해 주세요.", style = MaterialTheme.typography.bodySmall) }
        item { Text("같은 기록이 여러 주제에 나타날 수 있습니다. 전체 기록 수는 중복 없이 " + state.records.size + "개입니다.", style = MaterialTheme.typography.bodySmall) }
        if (state.computing || state.records.any { it.processing != "ready" }) item {
            Text("처리 대기·실패 기록이 있어 자동 군집을 아직 전체 결과로 표시하지 않습니다.", color = MaterialTheme.colorScheme.error)
        }
        if (state.topics.isEmpty()) item { Text("처리된 생활 기록을 추가하면 주제가 나타납니다.") }
        items(state.topics, key = { "topic-" + groupKey(it) }) { group ->
            LifeGroupCard(group, if (group.automatic) "자동 주제" else "사용자 주제") { onTopic(group) }
        }
    }
}

@Composable
private fun LifeTopicPage(
    state: LifeUiState, topic: LifeDisplayGroup?, onBack: () -> Unit, onRename: (LifeDisplayGroup) -> Unit,
    onOrder: (LifeDisplayGroup) -> Unit, onRecord: (LifeRecord) -> Unit,
) {
    val ids = topic?.recordIds?.toSet().orEmpty()
    val orders = state.orders.filter { group -> group.recordIds.any { it in ids } }
    val byId = state.records.associateBy(LifeRecord::id)
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 주제 목록") } }
        if (topic == null) item { Text("결과가 바뀌었습니다. 주제를 다시 선택해 주세요.") }
        else {
            item {
                Text(topic.title, style = MaterialTheme.typography.headlineSmall)
                Text(topic.recordIds.size.toString() + "개 기록")
                TextButton(onClick = { onRename(topic) }) { Text("주제 이름 수정") }
            }
            item { Text("주문과 연결 후보", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
            items(orders, key = { "topic-order-" + groupKey(it) }) { group ->
                val count = group.recordIds.count { it in ids }
                ElevatedCard(Modifier.fillMaxWidth().clickable { onOrder(group) }) {
                    Column(Modifier.padding(14.dp)) {
                        Text(group.title, fontWeight = FontWeight.SemiBold)
                        Text((if (group.automatic && group.recordIds.size == 1) "연결 확인 필요" else "주문") + " · 이 주제의 기록 " + count + "개")
                    }
                }
            }
            item { Text("이 주제의 기록", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
            items(topic.recordIds, key = { "topic-record-" + it }) { id ->
                byId[id]?.let { record -> LifeRecordCard(record) { onRecord(record) } }
            }
        }
    }
}

@Composable
private fun LifeOrderPage(
    state: LifeUiState, order: LifeDisplayGroup?, topic: LifeDisplayGroup?, onBack: () -> Unit,
    onRename: (LifeDisplayGroup) -> Unit, onRecord: (LifeRecord) -> Unit,
) {
    val byId = state.records.associateBy(LifeRecord::id)
    val ids = order?.recordIds?.filter { topic == null || it in topic.recordIds }.orEmpty()
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 이전 화면") } }
        if (order == null) item { Text("결과가 바뀌었습니다. 주문을 다시 선택해 주세요.") }
        else {
            item {
                Text(order.title, style = MaterialTheme.typography.headlineSmall)
                Text(ids.size.toString() + "개 기록")
                if (order.automatic && order.recordIds.size == 1) Text("연결할 주문인지 원문을 확인하세요.", style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = { onRename(order) }) { Text("주문 이름 수정") }
            }
            items(ids, key = { "order-record-" + it }) { id ->
                byId[id]?.let { record -> LifeRecordCard(record) { onRecord(record) } }
            }
        }
    }
}

@Composable
private fun LifeRecordPage(
    state: LifeUiState, record: LifeRecord?, onBack: () -> Unit, onEdit: (LifeRecord) -> Unit,
    onDelete: (LifeRecord) -> Unit, onChoose: (String) -> Unit, onSplit: (String) -> Unit,
    onTruth: (String) -> Unit, onRemoveTopic: (String) -> Unit,
) {
    var truth by remember(record?.id, record?.truthOrder) { mutableStateOf(record?.truthOrder.orEmpty()) }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 이전 화면") } }
        if (record == null) item { Text("기록이 삭제되었거나 보관 기간이 끝났습니다.") }
        else {
            item {
                Text(record.title, style = MaterialTheme.typography.headlineSmall)
                Text("출처 · " + (record.sourceName ?: when (record.kind) {
                    "memo" -> "직접 작성"; "share" -> "공유한 글"; "document" -> "선택한 문서"; else -> "알림"
                }), style = MaterialTheme.typography.bodySmall)
                Text("처리 상태 · " + record.processing, style = MaterialTheme.typography.bodySmall)
            }
            if (record.kind == "notification") {
                item { Text("알림 이력", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
                items(state.versions.filter { it.recordId == record.id }, key = { "version-" + it.id }) { version ->
                    ElevatedCard(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(lifeDate(version.postedAt), style = MaterialTheme.typography.labelSmall)
                            Text(version.title, fontWeight = FontWeight.SemiBold)
                            Text(version.text)
                        }
                    }
                }
            } else item { Text(record.text, style = MaterialTheme.typography.bodyMedium) }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("주제와 주문 정정", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    OutlinedButton(onClick = { onChoose("topic") }, modifier = Modifier.fillMaxWidth()) { Text("대표 주제 옮기기") }
                    OutlinedButton(onClick = { onChoose("extra") }, modifier = Modifier.fillMaxWidth()) { Text("추가 주제 후보 확인") }
                    OutlinedButton(onClick = { onChoose("order") }, modifier = Modifier.fillMaxWidth()) { Text("주문 옮기기·후보 확인") }
                    TextButton(onClick = { onSplit("order") }) { Text("다른 주문으로 분리") }
                    state.topics.filter { !it.automatic && record.id in it.recordIds }.forEach { group ->
                        TextButton(onClick = { onRemoveTopic(requireNotNull(group.id)) }) { Text("주제 연결 해제 · " + group.title) }
                    }
                }
            }
            item {
                OutlinedTextField(truth, { truth = it }, Modifier.fillMaxWidth(), label = { Text("평가용 정답 주문 (선택)") })
                Text("같은 주문에는 같은 이름을 입력합니다. 자동 묶음과 사용자의 정정과는 별도입니다.", style = MaterialTheme.typography.bodySmall)
                TextButton(onClick = { onTruth(truth) }) { Text("정답 저장") }
            }
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (record.kind in setOf("memo", "share")) TextButton(onClick = { onEdit(record) }) { Text("수정") }
                    TextButton(onClick = { onDelete(record) }) { Text("생활 기록에서 삭제", color = MaterialTheme.colorScheme.error) }
                }
            }
        }
    }
}

@Composable
private fun LifeTasksPage(
    state: LifeUiState, onBack: () -> Unit, onNew: () -> Unit, onCandidate: (LifeCandidate) -> Unit,
    onDismiss: (Long) -> Unit, onEdit: (LifeTask) -> Unit, onDone: (String, Boolean) -> Unit,
    onDelete: (String) -> Unit, onSource: (String) -> Unit,
) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("일정·할 일", style = MaterialTheme.typography.headlineSmall) }
        item { Text("후보는 확인 후 저장됩니다. 기기 캘린더나 알림은 만들지 않습니다.", style = MaterialTheme.typography.bodySmall) }
        item { Button(onClick = onNew) { Text("직접 추가") } }
        item { Text("확인할 후보 " + state.candidates.size + "개", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        items(state.candidates, key = { "candidate-" + it.id }) { candidate ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text((if (candidate.kind == "event") "일정 후보 · " else "할 일 후보 · ") + candidate.title)
                    Text("날짜 · " + (candidate.dueDate ?: "확인 필요"), style = MaterialTheme.typography.bodySmall)
                    Row {
                        TextButton(onClick = { onCandidate(candidate) }) { Text("확인·저장") }
                        TextButton(onClick = { onDismiss(candidate.id) }) { Text("제외") }
                        TextButton(onClick = { onSource(candidate.recordId) }) { Text("원문") }
                    }
                }
            }
        }
        item { Text("저장한 일정과 할 일", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        items(state.tasks, key = { "saved-task-" + it.id }) { task ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Row {
                        Checkbox(checked = task.done, onCheckedChange = { onDone(task.id, it) })
                        Column(Modifier.weight(1f)) {
                            Text((if (task.kind == "event") "일정 · " else "할 일 · ") + task.title)
                            Text(task.dueDate ?: "날짜 미정", style = MaterialTheme.typography.bodySmall)
                            if (task.sourceExpired) Text("원문 보관 기간이 끝났거나 삭제됐습니다.", style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    Row {
                        TextButton(onClick = { onEdit(task) }) { Text("수정") }
                        TextButton(onClick = { onDelete(task.id) }) { Text("삭제") }
                        if (!task.sourceExpired && task.sourceId != null) TextButton(onClick = { onSource(task.sourceId) }) { Text("원문") }
                    }
                }
            }
        }
    }
}

@Composable
private fun LifeSettingsPage(
    state: LifeUiState, onBack: () -> Unit, onPermission: () -> Unit, onAllowed: (String, Boolean) -> Unit,
    onCollect: (Boolean) -> Unit, onReload: () -> Unit, onClear: () -> Unit, onRetry: () -> Unit,
) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("알림 수집 설정", style = MaterialTheme.typography.headlineSmall) }
        item { Text("Android 알림 접근을 허용하면 이 앱이 알림을 받을 수 있습니다. 선택한 앱의 글만 저장합니다. 일부 알림은 내용이 가려지거나 수집이 지연될 수 있습니다.", style = MaterialTheme.typography.bodySmall) }
        item { Text("권한 · " + if (state.permissionGranted) "허용됨" else "허용되지 않음") }
        item { Text("연결 · " + if (state.connected) "연결됨" else "연결되지 않음") }
        item { OutlinedButton(onClick = onPermission) { Text("Android 알림 접근 설정") } }
        item {
            Row {
                Switch(checked = state.collecting, onCheckedChange = onCollect)
                Text("  " + when {
                    !state.collecting -> "알림 수집 꺼짐"
                    !state.permissionGranted -> "수집 설정 켜짐 · 권한 필요"
                    !state.connected -> "수집 설정 켜짐 · 연결 대기"
                    else -> "선택한 앱의 알림 수집 중"
                }, Modifier.padding(top = 12.dp))
            }
        }
        item { Text("수집할 앱", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        item { Text("앱이 보이지 않으면 해당 앱의 알림을 표시한 뒤 목록을 새로 고쳐 보세요.", style = MaterialTheme.typography.bodySmall) }
        item { TextButton(onClick = onReload) { Text("앱 목록 새로 고침") } }
        item { Text("크기 제한 초과·읽기 실패 등 제외된 알림 처리 · " + state.excluded + "회", style = MaterialTheme.typography.bodySmall) }
        item { OutlinedButton(onClick = onRetry) { Text("처리 실패 기록 다시 시도") } }
        item { OutlinedButton(onClick = onClear) { Text("수집한 알림 전체 삭제") } }
        items(state.apps, key = LifeAppChoice::packageName) { app ->
            Row(Modifier.fillMaxWidth().clickable { onAllowed(app.packageName, !app.selected) }) {
                Checkbox(checked = app.selected, onCheckedChange = { onAllowed(app.packageName, it) })
                Column(Modifier.padding(top = 8.dp)) {
                    Text(app.label)
                    Text(app.packageName, style = MaterialTheme.typography.labelSmall)
                }
            }
        }
    }
}

@Composable
private fun LifeTextEditor(
    record: LifeRecord?, shared: SharedLifeText?, onDismiss: () -> Unit,
    onSave: (String, String, (Boolean) -> Unit) -> Unit,
) {
    var title by rememberSaveable(record?.id, shared?.body) { mutableStateOf(record?.title ?: shared?.title.orEmpty()) }
    var body by rememberSaveable(record?.id, shared?.body) { mutableStateOf(record?.text ?: shared?.body.orEmpty()) }
    var error by remember { mutableStateOf<String?>(null) }
    var saving by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = { if (!saving) onDismiss() },
        title = { Text(if (record != null) "생활 기록 수정" else if (shared != null) "공유한 글 저장" else "메모 작성") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(title, { title = it }, Modifier.fillMaxWidth(), label = { Text("제목") })
                OutlinedTextField(body, { body = it }, Modifier.fillMaxWidth(), label = { Text("내용") }, minLines = 5)
                Text("20,000자까지 저장합니다. 공유한 글의 작성일은 자동으로 추정하지 않습니다.", style = MaterialTheme.typography.bodySmall)
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = {
            TextButton(enabled = !saving, onClick = {
                error = when {
                    title.isBlank() || body.isBlank() -> "제목과 내용을 입력해 주세요."
                    body.trim().length > LifeRepository.MAX_TEXT -> "내용이 20,000자를 넘습니다."
                    else -> null
                }
                if (error == null) { saving = true; onSave(title, body) { saving = false } }
            }) { Text("저장") }
        },
        dismissButton = { TextButton(enabled = !saving, onClick = onDismiss) { Text("취소") } },
    )
}

@Composable
private fun LifeTaskEditor(
    initialKind: String, initialTitle: String, initialDate: String?, onDismiss: () -> Unit,
    onSave: (String, String, String?, (Boolean) -> Unit) -> Unit,
) {
    var kind by rememberSaveable(initialTitle) { mutableStateOf(initialKind) }
    var title by rememberSaveable(initialTitle) { mutableStateOf(initialTitle) }
    var date by rememberSaveable(initialTitle) { mutableStateOf(initialDate.orEmpty()) }
    var error by remember { mutableStateOf<String?>(null) }
    var saving by remember { mutableStateOf(false) }
    AlertDialog(
        onDismissRequest = { if (!saving) onDismiss() },
        title = { Text("일정·할 일 확인") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = kind == "task", onClick = { kind = "task" }, label = { Text("할 일") })
                    FilterChip(selected = kind == "event", onClick = { kind = "event" }, label = { Text("일정") })
                }
                OutlinedTextField(title, { title = it }, Modifier.fillMaxWidth(), label = { Text("제목") })
                OutlinedTextField(date, { date = it }, Modifier.fillMaxWidth(), label = { Text("날짜 YYYY-MM-DD (선택)") }, singleLine = true)
                Text("날짜가 불확실하면 비워 두고 나중에 정할 수 있습니다.", style = MaterialTheme.typography.bodySmall)
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = {
            TextButton(enabled = !saving, onClick = {
                error = when {
                    title.isBlank() -> "제목을 입력해 주세요."
                    date.isNotBlank() && runCatching { LocalDate.parse(date.trim()) }.isFailure -> "날짜는 YYYY-MM-DD 형식으로 입력해 주세요."
                    else -> null
                }
                if (error == null) { saving = true; onSave(kind, title, date.trim().takeIf(String::isNotEmpty)) { saving = false } }
            }) { Text("저장") }
        },
        dismissButton = { TextButton(enabled = !saving, onClick = onDismiss) { Text("취소") } },
    )
}

@Composable
private fun LifeNameDialog(initial: String, onDismiss: () -> Unit, onSave: (String) -> Unit) {
    var name by rememberSaveable(initial) { mutableStateOf(initial) }
    AlertDialog(onDismissRequest = onDismiss, title = { Text("묶음 이름") },
        text = { OutlinedTextField(name, { name = it }, label = { Text("이름") }) },
        confirmButton = { TextButton(enabled = name.isNotBlank(), onClick = { onSave(name) }) { Text("저장") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("취소") } })
}

@Composable
private fun LifeGroupChoiceDialog(
    record: LifeRecord, kind: String, extra: Boolean, groups: List<LifeDisplayGroup>, suggestions: List<LifeDisplayGroup>,
    onDismiss: () -> Unit, onChoose: (LifeDisplayGroup) -> Unit, onCreate: (String) -> Unit,
) {
    var name by rememberSaveable(record.id, kind, extra) { mutableStateOf("") }
    val suggestionKeys = suggestions.map(::groupKey).toSet()
    val available = groups.filter { record.id !in it.recordIds }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (kind == "order") "주문 선택" else if (extra) "추가 주제 선택" else "대표 주제 선택") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("추천은 확인이 필요한 후보입니다.", style = MaterialTheme.typography.bodySmall)
                LazyColumn(Modifier.fillMaxWidth().heightIn(max = 260.dp)) {
                    items(available, key = ::groupKey) { group ->
                        TextButton(onClick = { onChoose(group) }) {
                            Text((if (groupKey(group) in suggestionKeys) "추천 · " else "") + group.title + " · " + group.recordIds.size + "개")
                        }
                    }
                }
                OutlinedTextField(name, { name = it }, Modifier.fillMaxWidth(), label = { Text(if (kind == "order") "새 주문 이름" else "새 주제 이름") })
                OutlinedButton(enabled = name.isNotBlank(), onClick = { onCreate(name.trim()) }) { Text("새 묶음 만들기") }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("닫기") } },
    )
}

@Composable
private fun LifeEvaluationPage(
    state: LifeUiState, viewModel: LifeViewModel, onBack: () -> Unit, onReview: (String) -> Unit,
    onRecord: (LifeRecord) -> Unit,
) {
    var showAutomatic by rememberSaveable { mutableStateOf(false) }
    val byId = state.records.associateBy(LifeRecord::id)
    val devCount = LifeSynthetic.orders(LifeSynthetic.DEV).size
    val testCount = LifeSynthetic.orders(LifeSynthetic.TEST).size
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("생활 군집 비교", style = MaterialTheme.typography.headlineSmall) }
        item { Text("가상 개발 자료와 별도 평가 자료를 사람이 확인합니다. 실제 기록은 기기에 남으며 가상 점수와 따로 보고합니다.", style = MaterialTheme.typography.bodySmall) }
        item {
            Text("1. 개발 자료 확인 · " + state.devReviewed.size + "/" + devCount + "개 주문")
            OutlinedButton(onClick = { onReview("dev-review") }) { Text("개발 자료 검토") }
            Button(onClick = viewModel::calibrate, enabled = state.devReviewed.size == devCount && state.evaluationProgress == null &&
                (state.trialStart == null || System.currentTimeMillis() >= state.trialStart + 7L * 24 * 60 * 60 * 1000)) {
                Text("개발 자료로 기준값 정하기")
            }
            Text("주문 기준 · " + (state.threshold?.let { "%.2f".format(Locale.ROOT, it) } ?: "미설정") +
                " · 주제 기준 · " + (state.topicThreshold?.let { "%.2f".format(Locale.ROOT, it) } ?: "미설정"), style = MaterialTheme.typography.bodySmall)
        }
        item {
            Text("2. 별도 평가 자료 확인 · " + state.testReviewed.size + "/" + testCount + "개 주문")
            OutlinedButton(onClick = { onReview("test-review") }) { Text("평가 자료 검토") }
            Button(onClick = viewModel::evaluateSynthetic, enabled = state.threshold != null && state.testReviewed.size == testCount && state.evaluationProgress == null) {
                Text("두 방식 평가")
            }
        }
        state.evaluationProgress?.let { progress -> item {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(progress)
            TextButton(onClick = viewModel::cancelEvaluation) { Text("취소") }
        } }
        state.evaluation?.let { result -> item {
            val current = state.threshold != null && kotlin.math.abs(state.threshold - result.threshold) < 0.0001 &&
                state.testReviewed.size == testCount && result.modelHash == state.modelHash && result.dataset == LifeSynthetic.TEST
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(if (current) "가상 평가 결과" else "이전 가상 평가 결과 · 현재 기준과 별도", fontWeight = FontWeight.SemiBold)
                    Text("관련 쌍 " + result.positives + " · 무관 쌍 " + result.negatives)
                    Text("모델 단독 F1 " + "%.3f".format(Locale.ROOT, result.modelF1) +
                        " · 정밀도 " + "%.3f".format(Locale.ROOT, result.modelPrecision) +
                        " · 재현율 " + "%.3f".format(Locale.ROOT, result.modelRecall) +
                        " · 잘못 합침 " + result.modelFalseMerges + "쌍")
                    Text(if (current && result.modelF1 >= 0.8) "모델 단독 목표 도달" else if (current) "모델 단독 목표 미달" else "현재 기준 미판정")
                    Text("단서 결합 F1 " + "%.3f".format(Locale.ROOT, result.hybridF1) +
                        " · 정밀도 " + "%.3f".format(Locale.ROOT, result.hybridPrecision) +
                        " · 재현율 " + "%.3f".format(Locale.ROOT, result.hybridRecall) +
                        " · 잘못 합침 " + result.hybridFalseMerges + "쌍")
                    Text(if (current && result.hybridF1 >= 0.8) "단서 결합 목표 도달" else if (current) "단서 결합 목표 미달" else "현재 기준 미판정")
                    Text("가상 자료의 결과입니다. 실제 7일 사용의 품질을 보장하지 않습니다.", style = MaterialTheme.typography.bodySmall)
                }
            }
        } }
        item {
            Text("3. 실제 생활 기록", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("기록 상세에서 평가용 정답 주문을 입력한 항목만 비교합니다. 자동 결과와 직접 고친 묶음은 따로 유지합니다.", style = MaterialTheme.typography.bodySmall)
            val metrics = state.realModelMetrics
            if (metrics == null || metrics.f1 == null) Text("정답을 입력한 관련·무관 기록 쌍이 부족해 미판정입니다.")
            else {
                Text("정답 기록 " + metrics.labeledRecords + "개 · 관련 " + metrics.positivePairs + "쌍 · 무관 " + metrics.negativePairs + "쌍")
                Text("모델 단독 F1 " + "%.3f".format(Locale.ROOT, metrics.f1) + " · 잘못 합침 " + metrics.falsePositive + "쌍")
                state.realHybridMetrics?.f1?.let { score ->
                    Text("단서 결합 F1 " + "%.3f".format(Locale.ROOT, score) + " · 잘못 합침 " + state.realHybridMetrics.falsePositive + "쌍")
                }
                if (metrics.positivePairs < 50 || metrics.negativePairs < 50 || state.threshold == null) {
                    Text("자료 또는 보정이 부족해 위 수치는 참고용이며 미판정입니다.", color = MaterialTheme.colorScheme.error)
                }
            }
            Button(onClick = { viewModel.saveComparison(); showAutomatic = true },
                enabled = !state.computing && state.records.isNotEmpty() && state.records.all { it.clusterVector != null } && state.threshold != null) {
                Text("현재 자동 결과 기록·비교")
            }
            TextButton(onClick = { showAutomatic = !showAutomatic }) { Text(if (showAutomatic) "자동 결과 접기" else "자동 결과 보기") }
        }
        if (showAutomatic) {
            item { Text("모델 단독", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold) }
            items(state.modelGroups.size, key = { "model-group-" + it }) { index ->
                val ids = state.modelGroups[index]
                ElevatedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text("자동 묶음 " + (index + 1) + " · " + ids.size + "개")
                        ids.forEach { id -> byId[id]?.let { record -> TextButton(onClick = { onRecord(record) }) { Text(record.title) } } }
                    }
                }
            }
            item { Text("단서 결합", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold) }
            items(state.hybridGroups.size, key = { "hybrid-group-" + it }) { index ->
                val ids = state.hybridGroups[index]
                ElevatedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(12.dp)) {
                        Text("자동 묶음 " + (index + 1) + " · " + ids.size + "개")
                        ids.forEach { id -> byId[id]?.let { record -> TextButton(onClick = { onRecord(record) }) { Text(record.title) } } }
                    }
                }
            }
        }
    }
}

@Composable
private fun LifeReviewPage(dataset: String, state: LifeUiState, viewModel: LifeViewModel, onBack: () -> Unit) {
    val orders = LifeSynthetic.orders(dataset).entries.toList()
    val reviewed = if (dataset == LifeSynthetic.DEV) state.devReviewed else state.testReviewed
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 군집 비교") } }
        item { Text(if (dataset == LifeSynthetic.DEV) "가상 개발 자료 검토" else "가상 평가 자료 검토", style = MaterialTheme.typography.headlineSmall) }
        item { Text("각 카드의 세 기록이 같은 주문이고 표시된 주제가 어울리는지 확인하세요. 서로 다른 주문 카드는 다른 주문입니다. 잘못된 카드가 있으면 확인하지 말고 자료를 수정해야 합니다.", style = MaterialTheme.typography.bodySmall) }
        items(orders, key = { it.key }) { entry ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("가상 주문 · " + entry.key + " · 주제 " + entry.value.first().topic, fontWeight = FontWeight.SemiBold)
                    entry.value.forEach { item -> Text(item.title + "\n" + item.text, style = MaterialTheme.typography.bodySmall) }
                    Row {
                        Checkbox(checked = entry.key in reviewed, onCheckedChange = { viewModel.setReviewed(dataset, entry.key, it) })
                        Text("같은 주문과 주제를 확인함", Modifier.padding(top = 12.dp))
                    }
                }
            }
        }
    }
}

@Composable
private fun LifeTrialPage(state: LifeUiState, viewModel: LifeViewModel, onBack: () -> Unit) {
    var scenario by rememberSaveable { mutableStateOf("기록 찾기") }
    var method by rememberSaveable { mutableStateOf("생활 POC") }
    var success by rememberSaveable { mutableStateOf(true) }
    var seconds by rememberSaveable { mutableStateOf("") }
    var issue by rememberSaveable { mutableStateOf("") }
    var error by remember { mutableStateOf<String?>(null) }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { TextButton(onClick = onBack) { Text("← 생활 화면") } }
        item { Text("7일 사용 기록", style = MaterialTheme.typography.headlineSmall) }
        item {
            Text(if (state.trialStart == null) "개발 자료 보정 뒤 7일 시험을 시작할 수 있습니다."
                else "시작 · " + lifeDate(state.trialStart) + " · 종료 예정 · " + lifeDate(state.trialStart + 7L * 24 * 60 * 60 * 1000))
            if (state.trialStart == null) Button(onClick = viewModel::startTrial, enabled = state.threshold != null) { Text("7일 시험 시작") }
            Text("기록 찾기·일정·알림 정리를 실제로 사용하며 성공·시간·불편을 남깁니다. 개인 원문은 적지 않습니다.", style = MaterialTheme.typography.bodySmall)
        }
        item {
            Text("시나리오")
            listOf("기록 찾기", "일정·할 일", "알림 정리").forEach { value ->
                FilterChip(selected = scenario == value, onClick = { scenario = value }, label = { Text(value) })
            }
            Text("사용한 방법")
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf("생활 POC", "평소 방식").forEach { value ->
                    FilterChip(selected = method == value, onClick = { method = value }, label = { Text(value) })
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(selected = success, onClick = { success = true }, label = { Text("성공") })
                FilterChip(selected = !success, onClick = { success = false }, label = { Text("실패") })
            }
            OutlinedTextField(seconds, { seconds = it }, Modifier.fillMaxWidth(), label = { Text("소요 시간(초, 선택)") }, singleLine = true)
            OutlinedTextField(issue, { issue = it }, Modifier.fillMaxWidth(), label = { Text("불편·오류(개인 원문 제외)") })
            error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            Button(onClick = {
                val duration = seconds.trim().takeIf(String::isNotEmpty)?.toIntOrNull()
                error = if (seconds.isNotBlank() && (duration == null || duration < 0)) "소요 시간은 0 이상의 초로 입력해 주세요." else null
                if (error == null) { viewModel.logUsage(scenario, method, success, duration, issue); seconds = ""; issue = "" }
            }) { Text("사용 결과 기록") }
        }
        item { Text("기록한 결과", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold) }
        if (state.usage.isEmpty()) item { Text("아직 기록한 결과가 없습니다.") }
        items(state.usage.size) { index -> Text(state.usage[index], style = MaterialTheme.typography.bodySmall) }
    }
}
