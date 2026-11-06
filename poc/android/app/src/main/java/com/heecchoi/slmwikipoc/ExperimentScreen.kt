package com.heecchoi.slmwikipoc

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Environment
import android.provider.Settings
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private data class InputDraft(val id: String?, val kind: String, val title: String, val body: String) : java.io.Serializable

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ExperimentApp(viewModel: ExperimentViewModel) {
    val state = viewModel.state
    val context = LocalContext.current
    var page by rememberSaveable { mutableStateOf("input") }
    var groupId by rememberSaveable { mutableStateOf<Int?>(null) }
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    var editing by rememberSaveable { mutableStateOf<InputDraft?>(null) }
    var labelItem by remember { mutableStateOf<ExperimentItem?>(null) }
    var deleting by remember { mutableStateOf<ExperimentItem?>(null) }
    var clearAll by rememberSaveable { mutableStateOf(false) }
    var documentChoice by rememberSaveable { mutableStateOf(false) }
    var voiceOpen by remember { mutableStateOf(false) }
    var voiceActive by remember { mutableStateOf(false) }
    var voiceText by remember { mutableStateOf("") }
    var voiceStatus by remember { mutableStateOf("") }
    var voiceError by remember { mutableStateOf<String?>(null) }
    val speech = remember {
        OnDeviceSpeech(context.applicationContext,
            onStatus = { voiceStatus = it; voiceError = null },
            onPartial = { voiceText = it },
            onResult = { text -> voiceActive = false; voiceOpen = false; editing = InputDraft(null, "voice", "", text) },
            onError = { error -> voiceActive = false; voiceError = error },
        )
    }
    DisposableEffect(speech) { onDispose { speech.cancel() } }
    val micPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { allowed ->
        voiceOpen = true
        voiceText = ""
        voiceError = null
        if (allowed) { voiceActive = true; speech.start() } else {
            voiceActive = false
            voiceError = "마이크 권한이 없어 음성 입력을 시작하지 않았습니다. 직접 입력할 수 있습니다."
        }
    }
    val documents = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { viewModel.previewDocuments(it) }
    val scanPermission = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (Environment.isExternalStorageManager()) viewModel.scanDevice() else viewModel.permissionDenied()
    }
    fun startVoice() {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            voiceOpen = true
            voiceText = ""
            voiceError = null
            voiceActive = true
            speech.start()
        } else micPermission.launch(Manifest.permission.RECORD_AUDIO)
    }
    fun startScan() {
        if (Environment.isExternalStorageManager()) viewModel.scanDevice()
        else {
            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${context.packageName}"))
            try { scanPermission.launch(intent) }
            catch (_: android.content.ActivityNotFoundException) { scanPermission.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
        }
    }
    fun back() {
        if (groupId != null) groupId = null else page = "input"
    }
    BackHandler(enabled = page == "result") { back() }

    Scaffold(topBar = { TopAppBar(title = { Text("군집 실험") }) }) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            if (state.busy) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column(Modifier.weight(1f).padding(top = 8.dp)) {
                        Text(state.busyText ?: "처리 중입니다", style = MaterialTheme.typography.bodySmall)
                        state.progress?.let { Text("${it.first}/${it.second}", style = MaterialTheme.typography.labelSmall) }
                    }
                    TextButton(onClick = viewModel::cancelOperation) { Text("취소") }
                }
            }
            if (page == "input") ExperimentInputPage(state,
                onDocuments = { documentChoice = true },
                onText = { editing = InputDraft(null, "text", "", "") },
                onVoice = ::startVoice, onSelect = { selectedId = it.id }, onClear = { clearAll = true },
                onRun = { groupId = null; page = "result"; viewModel.runClustering() },
                onResult = { groupId = null; page = "result" },
            ) else ExperimentResultPage(state, groupId, onBack = ::back, onGroup = { groupId = it },
                onSelect = { selectedId = it.id }, onRun = viewModel::runClustering, onThreshold = viewModel::setThreshold)
        }
    }

    if (documentChoice) AlertDialog(onDismissRequest = { documentChoice = false }, title = { Text("문서 추가") },
        text = { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("파일을 직접 선택할 수 있습니다. 기기 스캔에는 모든 파일 접근 권한이 필요하며, 접근 가능한 공유 저장소만 검색합니다. 다른 앱의 비공개 파일과 클라우드 전용 파일은 검색하지 않습니다.")
            OutlinedButton(onClick = { documentChoice = false; documents.launch(arrayOf("*/*")) }, modifier = Modifier.fillMaxWidth()) { Text("파일 선택") }
            OutlinedButton(onClick = { documentChoice = false; startScan() }, modifier = Modifier.fillMaxWidth()) { Text("기기 문서 스캔") }
        } },
        confirmButton = { TextButton(onClick = { documentChoice = false }) { Text("닫기") } })

    state.preview?.let { preview ->
        DocumentImportPreviewDialog(preview, !state.busy, viewModel::setImportLabel, viewModel::cancelPreview, viewModel::confirmImport)
    }
    editing?.let { draft ->
        InputEditor(draft, onDismiss = { editing = null }, onSave = { title, body, done ->
            viewModel.saveText(draft.id, draft.kind, title, body) { ok -> if (ok) editing = null; done(ok) }
        })
    }
    val selected = state.items.firstOrNull { it.id == selectedId }
    selected?.let { item ->
        InputDetail(item, onDismiss = { selectedId = null }, onLabel = { selectedId = null; labelItem = item },
            onEdit = { selectedId = null; editing = InputDraft(item.id, item.kind, item.title, item.body) },
            onDelete = { selectedId = null; deleting = item })
    }
    labelItem?.let { item ->
        LabelEditor(item.label.orEmpty(), onDismiss = { labelItem = null }, onSave = { value, done ->
            viewModel.setLabel(item.id, value) { ok -> if (ok) labelItem = null; done(ok) }
        })
    }
    deleting?.let { item ->
        AlertDialog(onDismissRequest = { deleting = null }, title = { Text("실험 기록을 삭제할까요?") },
            text = { Text("이번 실험의 글과 임베딩을 삭제합니다. 기기 원본과 업데이트 전 기록은 유지합니다.") },
            confirmButton = { TextButton(onClick = { deleting = null; viewModel.remove(item.id) }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } })
    }
    if (clearAll) AlertDialog(onDismissRequest = { clearAll = false }, title = { Text("실험 목록을 비울까요?") },
        text = { Text("이번 실험의 글·임베딩·결과를 삭제합니다. 기기 원본과 업데이트 전 기록은 유지합니다.") },
        confirmButton = { TextButton(onClick = { clearAll = false; viewModel.clear() }) { Text("비우기") } },
        dismissButton = { TextButton(onClick = { clearAll = false }) { Text("취소") } })
    if (voiceOpen) AlertDialog(onDismissRequest = { speech.cancel(); voiceOpen = false }, title = { Text("음성 입력") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(voiceError ?: voiceStatus)
                if (voiceText.isNotBlank()) Text(voiceText, style = MaterialTheme.typography.bodySmall)
                Text("기기 내 한국어 음성 인식만 사용합니다. 인식 결과는 저장 전에 수정할 수 있습니다.", style = MaterialTheme.typography.bodySmall)
            }
        },
        confirmButton = {
            if (voiceActive) TextButton(onClick = { voiceActive = false; voiceStatus = "음성을 글로 바꾸는 중입니다."; speech.stop() }) { Text("말하기 완료") }
            else if (voiceError != null) TextButton(onClick = {
                voiceError = null; voiceText = ""
                if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
                    voiceActive = true; speech.start()
                } else micPermission.launch(Manifest.permission.RECORD_AUDIO)
            }) { Text("다시 시도") }
        },
        dismissButton = { TextButton(onClick = { speech.cancel(); voiceOpen = false }) { Text("취소") } })
    state.message?.let { message ->
        AlertDialog(onDismissRequest = viewModel::clearMessage, title = { Text("군집 실험") }, text = { Text(message) },
            confirmButton = { TextButton(onClick = viewModel::clearMessage) { Text("확인") } })
    }
}

@Composable
private fun ExperimentInputPage(
    state: ExperimentUiState, onDocuments: () -> Unit, onText: () -> Unit, onVoice: () -> Unit,
    onSelect: (ExperimentItem) -> Unit, onClear: () -> Unit, onRun: () -> Unit, onResult: () -> Unit,
) {
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text("실험할 글 추가", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold) }
        item { Text("문서, 직접 입력, 음성으로 넣은 글을 함께 묶습니다.", style = MaterialTheme.typography.bodySmall) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedButton(onClick = onDocuments, enabled = !state.busy, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 4.dp)) { Text("문서") }
                OutlinedButton(onClick = onText, enabled = !state.busy, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 4.dp)) { Text("직접 입력") }
                OutlinedButton(onClick = onVoice, enabled = !state.busy, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 4.dp)) { Text("음성") }
            }
        }
        state.report?.let { report -> item {
            ElevatedCard(Modifier.fillMaxWidth()) { Text(report, Modifier.padding(14.dp), style = MaterialTheme.typography.bodySmall) }
        } }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Button(onClick = onRun, enabled = !state.busy && state.items.isNotEmpty(), modifier = Modifier.fillMaxWidth()) { Text("군집 실행") }
                if (state.run != null) OutlinedButton(onClick = onResult, modifier = Modifier.fillMaxWidth()) { Text("마지막 결과 보기") }
            }
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("실험 기록 ${state.items.size}개", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                if (state.items.isNotEmpty()) TextButton(onClick = onClear, enabled = !state.busy) { Text("목록 비우기") }
            }
        }
        if (state.items.isEmpty()) item { Text("문서를 가져오거나 글을 입력해 주세요.", style = MaterialTheme.typography.bodySmall) }
        items(state.items, key = ExperimentItem::id) { item ->
            ElevatedCard(Modifier.fillMaxWidth().clickable(enabled = !state.busy) { onSelect(item) }) {
                Column(Modifier.padding(15.dp)) {
                    Text(item.title, fontWeight = FontWeight.SemiBold)
                    item.scanError?.let { Text("읽기 실패 · $it", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                }
            }
        }
    }
}

@Composable
private fun ExperimentResultPage(
    state: ExperimentUiState, selectedGroup: Int?, onBack: () -> Unit, onGroup: (Int?) -> Unit,
    onSelect: (ExperimentItem) -> Unit, onRun: () -> Unit, onThreshold: (Float) -> Unit,
) {
    var showEvaluation by rememberSaveable { mutableStateOf(false) }
    var slider by remember(state.threshold) { mutableFloatStateOf(state.threshold.toFloat()) }
    val run = state.run
    val byId = state.items.associateBy(ExperimentItem::id)
    val points = run?.points.orEmpty().filter { it.inputId in byId }
    val groupMap = points.groupBy(ExperimentRunPoint::groupId).toSortedMap()
    val groupNotes = groupMap.map { (id, members) -> NoteGroup(id, "군집", members.mapNotNull { byId[it.inputId]?.note() }) }
    val mapNotes = points.mapNotNull { byId[it.inputId]?.note() }
    val projected = points.map { ProjectedDocument(it.inputId, it.x, it.y) }
    val groupedIds = groupMap.filterValues { it.first().groupSize > 1 }.keys
    val multiGroups = groupNotes.filter { it.id in groupedIds }
    val singles = groupNotes.filter { it.id !in groupedIds }.flatMap(NoteGroup::notes)
    if (selectedGroup != null) {
        val selectedNotes = if (selectedGroup == -1) singles else groupNotes.firstOrNull { it.id == selectedGroup }?.notes.orEmpty()
        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            item { TextButton(onClick = { onGroup(null) }) { Text("← 군집 결과") } }
            if (selectedNotes.isEmpty()) item { Text("선택한 군집의 기록이 삭제되었습니다.") }
            else {
                item { Text(if (selectedGroup == -1) "미분류" else "군집 ${multiGroups.indexOfFirst { it.id == selectedGroup } + 1}", style = MaterialTheme.typography.headlineSmall) }
                item { Text("${selectedNotes.size}개 기록") }
            }
            if (state.stale) item { Text("이전 실행 결과입니다. 현재 입력과 다를 수 있습니다.", color = MaterialTheme.colorScheme.error) }
            items(selectedNotes, key = Note::id) { note ->
                byId[note.id]?.let { item ->
                    ElevatedCard(Modifier.fillMaxWidth().clickable { onSelect(item) }) { Text(item.title, Modifier.padding(16.dp), fontWeight = FontWeight.SemiBold) }
                }
            }
        }
        return
    }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { TextButton(onClick = onBack) { Text("← 입력 목록") } }
        item { Text("군집 결과", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold) }
        if (state.stale) item { Text("입력이나 기준이 바뀌었습니다. 아래는 이전 결과이며 다시 실행해야 현재 입력을 반영합니다.", color = MaterialTheme.colorScheme.error) }
        if (run == null) item { Text(if (state.busy) "처리가 끝나면 결과를 표시합니다." else "입력을 추가하고 군집을 실행해 주세요.") }
        if (run != null) {
            item { Text("마지막 실행 ${SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.KOREA).format(Date(run.createdAt))} · 거리 기준 ${"%.2f".format(Locale.ROOT, run.threshold)}", style = MaterialTheme.typography.bodySmall) }
            item { Text("${multiGroups.size}개 군집 · 군집 기록 ${multiGroups.sumOf { it.notes.size }}개 · 미분류 ${singles.size}개 · 결과 ${points.size}개 / 현재 입력 ${state.items.size}개", style = MaterialTheme.typography.bodySmall) }
            if (points.isNotEmpty()) item {
                Text("2D PCA 지도", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                PcaMap(projected, mapNotes, groupNotes, forceGroupedIds = groupedIds) { note -> byId[note.id]?.let(onSelect) }
                Text("지도는 임베딩을 2차원으로 표시합니다. 점 사이의 거리가 원래 거리와 정확히 같지는 않습니다.", style = MaterialTheme.typography.bodySmall)
            }
            items(multiGroups, key = { "group-${it.id}" }) { group ->
                val index = multiGroups.indexOf(group) + 1
                ElevatedCard(Modifier.fillMaxWidth().clickable { onGroup(group.id) }) {
                    Column(Modifier.padding(16.dp)) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            Canvas(Modifier.size(12.dp)) { drawCircle(clusterColor(groupNotes.indexOf(group), group, true)) }
                            Text("군집 $index", fontWeight = FontWeight.SemiBold)
                        }
                        Text("${group.notes.size}개 기록 · " + group.notes.take(2).joinToString(" · ") { it.title }, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            if (singles.isNotEmpty()) item {
                ElevatedCard(Modifier.fillMaxWidth().clickable { onGroup(-1) }) {
                    Column(Modifier.padding(16.dp)) { Text("미분류", fontWeight = FontWeight.SemiBold); Text("${singles.size}개 기록") }
                }
            }
        }
        item {
            TextButton(onClick = { showEvaluation = !showEvaluation }) { Text(if (showEvaluation) "평가 접기" else "평가와 거리 기준") }
            if (showEvaluation) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("거리 기준 ${"%.2f".format(Locale.ROOT, slider)}")
                    Slider(value = slider, onValueChange = { slider = it }, onValueChangeFinished = { onThreshold(slider) },
                        valueRange = 0f..2f, steps = 199, enabled = !state.busy)
                    Text("기준을 바꾼 뒤 군집을 다시 실행해 주세요.", style = MaterialTheme.typography.bodySmall)
                    run?.let {
                        Text("임베딩 ${it.embeddingMs}ms · 새 계산 ${it.computed}개 · 재사용 ${it.reused}개")
                        Text("군집 계산 ${it.clusteringMs}ms · 마지막 실행 기준")
                        if (state.stale) Text("입력이 바뀌어 품질 지표는 다시 실행한 뒤 표시합니다.")
                        else {
                            val quality = it.quality
                            Text("정답 주제 ${quality.labeled}/${state.items.size}개 · 평가 ${quality.pairs}쌍")
                            if (quality.f1 == null) Text("정답 주제가 있는 기록이 2개 이상 필요합니다.")
                            else Text("정밀도 ${percentValue(quality.precision)} · 재현율 ${percentValue(quality.recall)} · F1 ${percentValue(quality.f1)} · 잘못 합친 쌍 ${quality.falseMerges}개")
                        }
                    }
                }
            }
        }
        item { Button(onClick = onRun, enabled = !state.busy && state.items.isNotEmpty(), modifier = Modifier.fillMaxWidth()) { Text("군집 다시 실행") } }
    }
}

private fun percentValue(value: Double?): String = value?.let { "%.1f%%".format(Locale.ROOT, it * 100) } ?: "미판정"

@Composable
private fun InputEditor(draft: InputDraft, onDismiss: () -> Unit, onSave: (String, String, (Boolean) -> Unit) -> Unit) {
    var title by rememberSaveable(draft.id, draft.body) { mutableStateOf(draft.title) }
    var body by rememberSaveable(draft.id, draft.body) { mutableStateOf(draft.body) }
    var error by remember { mutableStateOf<String?>(null) }
    var saving by remember { mutableStateOf(false) }
    AlertDialog(onDismissRequest = { if (!saving) onDismiss() }, title = { Text(if (draft.id != null) "기록 수정" else if (draft.kind == "voice") "음성 인식 결과" else "직접 입력") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedTextField(title, { title = it }, Modifier.fillMaxWidth(), label = { Text("제목 (비우면 첫 줄 사용)") })
                OutlinedTextField(body, { body = it }, Modifier.fillMaxWidth(), label = { Text("내용") }, minLines = 5)
                Text("내용은 20,000자까지 저장합니다.", style = MaterialTheme.typography.bodySmall)
                error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            }
        },
        confirmButton = { TextButton(enabled = !saving, onClick = {
            error = when {
                body.isBlank() -> "내용을 입력해 주세요."
                body.trim().length > ExperimentStore.MAX_TEXT -> "20,000자를 넘는 내용은 저장할 수 없습니다."
                else -> null
            }
            if (error == null) { saving = true; onSave(title, body) { saving = false } }
        }) { Text("저장") } },
        dismissButton = { TextButton(enabled = !saving, onClick = onDismiss) { Text("취소") } })
}

@Composable
private fun InputDetail(item: ExperimentItem, onDismiss: () -> Unit, onLabel: () -> Unit, onEdit: () -> Unit, onDelete: () -> Unit) {
    AlertDialog(onDismissRequest = onDismiss, title = { Text(item.title) },
        text = {
            Column(Modifier.fillMaxWidth().heightIn(max = 480.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(item.body)
                Text("정답 주제 · " + (item.label ?: "미지정"), style = MaterialTheme.typography.bodySmall)
                item.scanError?.let { Text("읽기 실패 · $it", color = MaterialTheme.colorScheme.error) }
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = onLabel) { Text("정답 주제") }
                    if (item.kind == "text" || item.kind == "voice") TextButton(onClick = onEdit) { Text("수정") }
                    TextButton(onClick = onDelete) { Text("삭제") }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("닫기") } })
}

@Composable
private fun LabelEditor(initial: String, onDismiss: () -> Unit, onSave: (String, (Boolean) -> Unit) -> Unit) {
    var value by rememberSaveable(initial) { mutableStateOf(initial) }
    var saving by remember { mutableStateOf(false) }
    AlertDialog(onDismissRequest = { if (!saving) onDismiss() }, title = { Text("정답 주제") },
        text = { Column {
            Text("같은 주제의 기록에 같은 이름을 입력합니다. 비워 두면 품질 평가에서 제외합니다.", style = MaterialTheme.typography.bodySmall)
            OutlinedTextField(value, { value = it }, Modifier.fillMaxWidth(), label = { Text("정답 주제 (선택)") })
        } },
        confirmButton = { TextButton(enabled = !saving, onClick = { saving = true; onSave(value) { saving = false } }) { Text("저장") } },
        dismissButton = { TextButton(enabled = !saving, onClick = onDismiss) { Text("취소") } })
}
