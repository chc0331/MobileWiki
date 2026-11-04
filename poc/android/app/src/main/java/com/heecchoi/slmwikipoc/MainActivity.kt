package com.heecchoi.slmwikipoc

import android.os.Bundle
import android.os.Environment
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModelProvider
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class MainActivity : ComponentActivity() {
    private val viewModel: PocViewModel by lazy { ViewModelProvider(this)[PocViewModel::class.java] }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme { PocApp(viewModel) } }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PocApp(viewModel: PocViewModel) {
    val state = viewModel.state
    val context = LocalContext.current
    var editing by remember { mutableStateOf<Note?>(null) }
    var showEditor by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf<Note?>(null) }
    var editingLabel by remember { mutableStateOf<Note?>(null) }
    val reportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { viewModel.exportReport(it) }
    val documentLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { viewModel.previewDocuments(it) }
    val scanPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) {
        if (Environment.isExternalStorageManager()) viewModel.startDeviceScan() else viewModel.deviceScanPermissionDenied()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("문서 군집 실험", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(state.modelStatus, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar {
                listOf(AppTab.NOTES, AppTab.GROUPS).forEach { tab ->
                    NavigationBarItem(
                        selected = state.tab == tab,
                        onClick = { viewModel.selectTab(tab) },
                        icon = { Text(tab.label.take(1)) },
                        label = { Text(tab.label) },
                    )
                }
            }
        },
        floatingActionButton = {
            if (state.tab == AppTab.NOTES && state.workspace == "personal" && state.modelReady) {
                Button(onClick = { editing = null; showEditor = true }) { Text("＋ 메모 작성") }
            }
        },
    ) { padding ->
        Column(Modifier.fillMaxSize().padding(padding)) {
            if (state.loading) {
                LinearProgressIndicator(Modifier.fillMaxWidth())
                state.busyMessage?.let { Text(it, Modifier.padding(horizontal = 16.dp, vertical = 6.dp), style = MaterialTheme.typography.labelMedium) }
                state.scanProgress?.let { (done, total) -> Text("처리 $done/$total", Modifier.padding(horizontal = 16.dp), style = MaterialTheme.typography.labelMedium) }
                if (state.scanActive) TextButton(onClick = viewModel::cancelDeviceScan) { Text("스캔 취소") }
            }
            state.modelError?.let {
                Card(Modifier.fillMaxWidth().padding(12.dp)) {
                    Column(Modifier.padding(16.dp)) {
                        Text("학습 모델을 불러오지 못했습니다", fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.error)
                        Text(it, Modifier.padding(top = 6.dp))
                        Text("기본 모델로 대체하지 않았습니다. 이 오류가 해결되기 전에는 검색·군집·평가를 사용할 수 없습니다.", Modifier.padding(top = 6.dp), style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            state.message?.let { Text(it, Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp), color = MaterialTheme.colorScheme.primary, style = MaterialTheme.typography.bodySmall) }
            when (state.tab) {
                AppTab.NOTES -> NotesScreen(
                    viewModel,
                    onEdit = { editing = it; showEditor = true },
                    onDelete = { deleting = it },
                    onEditLabel = { editingLabel = it },
                    onImportDocuments = { documentLauncher.launch(arrayOf("*/*")) },
                    onScanDevice = {
                        if (Environment.isExternalStorageManager()) viewModel.startDeviceScan()
                        else {
                            val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION, Uri.parse("package:${context.packageName}"))
                            try { scanPermissionLauncher.launch(intent) }
                            catch (_: android.content.ActivityNotFoundException) { scanPermissionLauncher.launch(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)) }
                        }
                    },
                )
                AppTab.GROUPS -> GroupsScreen(viewModel)
                AppTab.SEARCH -> SearchScreen(viewModel)
                AppTab.EVALUATION -> EvaluationScreen(viewModel) { reportLauncher.launch("embedding-poc-report.json") }
            }
        }
    }

    if (showEditor) NoteEditorDialog(editing, onDismiss = { showEditor = false }, onSave = { id, title, text -> viewModel.saveNote(id, title, text); showEditor = false })
    state.importPreview?.let { previews ->
        DocumentImportPreviewDialog(
            previews = previews,
            enabled = !state.loading,
            onLabelChange = viewModel::setImportGroundTruth,
            onDismiss = viewModel::cancelDocumentImport,
            onConfirm = viewModel::commitDocumentImport,
        )
    }
    editingLabel?.let { note ->
        GroundTruthLabelDialog(
            note = note,
            onDismiss = { editingLabel = null },
            onSave = { label -> viewModel.updateExperimentGroundTruth(note, label); editingLabel = null },
        )
    }
    deleting?.let { note ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text(if (note.workspace == "experiment") "문서를 삭제할까요?" else "메모를 삭제할까요?") },
            text = {
                Text(
                    if (note.source?.devicePath != null) "‘${note.title}’의 앱 내 추출 글과 임베딩을 삭제합니다. 기기의 원본은 그대로 두며 다음 스캔에서 다시 추가됩니다."
                    else if (note.source?.storedPath != null) "‘${note.title}’의 앱 내 문서 글과 PDF 원본, 임베딩을 삭제합니다."
                    else if (note.workspace == "experiment") "‘${note.title}’의 앱 내 문서 글과 임베딩을 삭제합니다."
                    else "‘${note.title}’ 메모와 이 메모의 로컬 임베딩을 삭제합니다."
                )
            },
            confirmButton = { TextButton(onClick = { viewModel.deleteNote(note); deleting = null }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } },
        )
    }
}

@Composable
private fun NotesScreen(
    viewModel: PocViewModel,
    onEdit: (Note) -> Unit,
    onDelete: (Note) -> Unit,
    onEditLabel: (Note) -> Unit,
    onImportDocuments: () -> Unit,
    onScanDevice: () -> Unit,
) {
    val state = viewModel.state
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { Text(
            if (state.workspace == "experiment") "실험 문서와 정답 주제는 이 기기의 앱 전용 저장소에만 저장합니다."
            else "개인 메모는 이 기기의 앱 전용 저장소에만 저장합니다.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        ) }
        if (state.workspace == "experiment") item {
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("가져온 문서를 개인 메모·합성 샘플과 분리해 군집합니다.", fontWeight = FontWeight.SemiBold)
                    Text("UTF-8 TXT·Markdown과 텍스트 PDF를 지원합니다. PDF 스캔본은 읽지 않습니다.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = onImportDocuments, enabled = !state.loading && state.modelReady) { Text("문서 가져오기") }
                    Text("기기 문서 스캔은 접근 가능한 공유 저장소를 검색합니다. 모든 파일 접근을 허용해야 하며 원본 파일은 옮기지 않습니다.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = onScanDevice, enabled = !state.loading && state.modelReady) { Text("기기 문서 스캔") }
                }
            }
        }
        if (state.workspace == "sample") item {
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("개인 메모와 분리된 합성 자료입니다.", fontWeight = FontWeight.SemiBold)
                    Text("고정 평가 데이터와 별도 공간에 저장합니다.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = viewModel::importSamples, enabled = !state.loading && state.modelReady) { Text("샘플 메모 가져오기") }
                }
            }
        }
        if (state.notes.isEmpty()) item {
            EmptyCard(when (state.workspace) {
                "personal" -> "아직 메모가 없습니다. 아래 버튼으로 메모를 작성해 보세요."
                "experiment" -> "실험할 문서를 가져오면 이 공간의 문서끼리 군집합니다."
                else -> "샘플이 없습니다. 위에서 합성 샘플을 가져오세요."
            })
        } else {
            item { Text("${state.notes.size}개 ${if (state.workspace == "experiment") "문서" else "메모"}", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold) }
            items(state.notes, key = Note::id) { note ->
                ElevatedCard(Modifier.fillMaxWidth().clickable { if (note.workspace == "personal") onEdit(note) }) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(note.title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(
                            note.text.take(if (note.workspace == "experiment") 1800 else Int.MAX_VALUE) + if (note.workspace == "experiment" && note.text.length > 1800) "…" else "",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        note.source?.let { source ->
                            Text("파일 · ${source.fileName}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                            source.devicePath?.let { Text("기기 스캔 · $it", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                        }
                        if (note.workspace == "experiment") Text("정답 주제 · ${note.groundTruthGroup ?: "미지정"}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text(formatDate(note.updatedAt), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (note.workspace == "personal") Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = { onEdit(note) }) { Text("수정") }
                            TextButton(onClick = { onDelete(note) }) { Text("삭제", color = MaterialTheme.colorScheme.error) }
                        }
                        if (note.workspace == "experiment") Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = { onEditLabel(note) }) { Text("정답 주제 수정") }
                            TextButton(onClick = { onDelete(note) }) { Text("삭제", color = MaterialTheme.colorScheme.error) }
                        }
                    }
                }
            }
        }
        item { Spacer(Modifier.height(72.dp)) }
    }
}

@Composable
private fun GroupsScreen(viewModel: PocViewModel) {
    val state = viewModel.state
    var selectedGroupId by rememberSaveable { mutableStateOf<Int?>(null) }
    var unclassifiedOpen by rememberSaveable { mutableStateOf(false) }
    var selectedDocumentId by rememberSaveable { mutableStateOf<String?>(null) }
    var previousGroups by remember { mutableStateOf(state.groups) }
    val grouped = state.groups.withIndex().filter { it.value.notes.size > 1 }
    val unclassified = state.groups.filter { it.notes.size == 1 }.flatMap { it.notes }
    val selectedGroup = grouped.firstOrNull { it.value.id == selectedGroupId }
    val selectedDocument = state.notes.firstOrNull { it.id == selectedDocumentId }
    LaunchedEffect(state.groups) {
        if (previousGroups !== state.groups) {
            selectedGroupId = null
            unclassifiedOpen = false
            selectedDocumentId = null
        }
        previousGroups = state.groups
    }
    LaunchedEffect(state.notes) {
        if (selectedDocumentId != null && state.notes.none { it.id == selectedDocumentId }) selectedDocumentId = null
    }
    BackHandler(enabled = state.workspace == "experiment" && (selectedDocumentId != null || selectedGroupId != null || unclassifiedOpen)) {
        if (selectedDocumentId != null) selectedDocumentId = null
        else {
            selectedGroupId = null
            unclassifiedOpen = false
        }
    }
    if (state.workspace == "experiment" && selectedDocument != null) {
        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item { TextButton(onClick = { selectedDocumentId = null }) { Text("← ${if (selectedGroup != null || unclassifiedOpen) "문서 목록" else "군집 개요"}") } }
            item { Text(selectedDocument.title, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold) }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    selectedDocument.source?.let { source ->
                        Text("파일 · ${source.fileName}", style = MaterialTheme.typography.bodySmall)
                        source.devicePath?.let { Text("기기 스캔 · $it", style = MaterialTheme.typography.bodySmall) }
                    }
                    Text("정답 주제 · ${selectedDocument.groundTruthGroup ?: "미지정"}", style = MaterialTheme.typography.bodySmall)
                    Text("갱신 · ${formatDate(selectedDocument.updatedAt)}", style = MaterialTheme.typography.bodySmall)
                }
            }
            item { Text("추출한 글", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold) }
            item { Text(selectedDocument.text, style = MaterialTheme.typography.bodyMedium) }
        }
        return
    }
    if (state.workspace == "experiment" && (selectedGroup != null || unclassifiedOpen)) {
        val documents = selectedGroup?.value?.notes ?: unclassified
        val groupTitle = if (unclassifiedOpen) "미분류" else "군집 ${grouped.indexOf(selectedGroup) + 1}"
        LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item {
                TextButton(onClick = { selectedGroupId = null; unclassifiedOpen = false }) { Text("← 군집 개요") }
            }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Canvas(Modifier.size(12.dp)) { drawCircle(if (selectedGroup == null) Color(0xFF8A8A8A) else clusterColor(selectedGroup.index, selectedGroup.value)) }
                        Text(groupTitle, style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
                    }
                    Text("${documents.size}개 문서", style = MaterialTheme.typography.titleSmall)
                    if (unclassifiedOpen) Text("각 문서가 1개짜리 군집에 있습니다.", style = MaterialTheme.typography.bodySmall)
                }
            }
            if (selectedGroup != null) item {
                val groupKey = selectedGroup.value.notes.map { it.id }.sorted().joinToString(",")
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = { viewModel.feedback("experiment_group", groupKey, true) }) { Text("👍 맞게 묶임") }
                    TextButton(onClick = { viewModel.feedback("experiment_group", groupKey, false) }) { Text("👎 잘못 묶임") }
                }
            }
            items(documents, key = Note::id) { note ->
                ElevatedCard(Modifier.fillMaxWidth().clickable { selectedDocumentId = note.id }) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(note.title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(note.text.take(160) + if (note.text.length > 160) "…" else "", style = MaterialTheme.typography.bodySmall, maxLines = 2)
                        Text("정답 주제 · ${note.groundTruthGroup ?: "미지정"}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Text("문서 열기 ›", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                        if (unclassifiedOpen) Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            TextButton(onClick = { viewModel.feedback("experiment_group", note.id, true) }) { Text("👍 맞게 분류됨") }
                            TextButton(onClick = { viewModel.feedback("experiment_group", note.id, false) }) { Text("👎 잘못 분류됨") }
                        }
                    }
                }
            }
        }
        return
    }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        state.scanReport?.let { report -> item {
            ElevatedCard(Modifier.fillMaxWidth()) {
                Text(report, Modifier.padding(16.dp), style = MaterialTheme.typography.bodySmall)
            }
        } }
        item { Text(
            when (state.workspace) {
                "personal" -> "개인 메모를 코사인 거리 평균 연결 방식으로 묶습니다. 고정 평가에서 정한 기준은 ${"%.2f".format(Locale.ROOT, state.clusterThreshold)}입니다."
                "experiment" -> "가져온 문서의 임베딩을 코사인 거리 평균 연결 방식으로 묶습니다."
                else -> "합성 샘플 메모를 코사인 거리 평균 연결 방식으로 묶습니다. 고정 평가 기준은 ${"%.2f".format(Locale.ROOT, state.clusterThreshold)}입니다."
            },
            style = MaterialTheme.typography.bodySmall,
        ) }
        if (state.workspace == "experiment") {
            item { ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("거리 임계값 ${"%.2f".format(Locale.ROOT, state.experimentThreshold)}", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                    Slider(
                        value = state.experimentThreshold.toFloat(),
                        onValueChange = viewModel::setExperimentThreshold,
                        onValueChangeFinished = viewModel::applyExperimentThreshold,
                        valueRange = 0f..2f,
                        steps = 199,
                        enabled = !state.loading && state.notes.size > 1,
                    )
                    Text("값이 커지면 더 많은 문서가 같은 군집에 들어갈 수 있습니다. 고정 평가 기준값에는 영향을 주지 않습니다.", style = MaterialTheme.typography.bodySmall)
                }
            } }
            if (state.groupCalculationActive) item {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                    Text("군집과 문서 수를 계산하는 중입니다.", style = MaterialTheme.typography.bodySmall)
                }
            } else {
            state.experimentMetrics?.let { metrics -> item {
                ElevatedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text("정답 라벨 기준 평가", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Text("라벨 문서 ${metrics.labeledDocumentCount}/${state.notes.size}개 · 문서 쌍 ${metrics.labeledPairCount}쌍 · 라벨 비율 ${percent(metrics.labelCoverage)}")
                        if (metrics.f1 == null) Text("라벨이 있는 문서가 2개 이상이면 지표를 계산합니다.", style = MaterialTheme.typography.bodySmall)
                        else {
                            Text("정밀도 ${percent(metrics.precision!!)} · 재현율 ${percent(metrics.recall!!)} · F1 ${percent(metrics.f1)}")
                            Text("정밀도는 한 군집에 묶인 문서 쌍 중 같은 주제인 비율, 재현율은 같은 주제 문서 쌍을 함께 묶은 비율입니다.", style = MaterialTheme.typography.bodySmall)
                            Text("다른 정답 주제를 합친 문서 쌍 ${metrics.falseMerges}개", style = MaterialTheme.typography.bodySmall)
                        }
                        Text("라벨이 없는 문서는 군집에는 포함하지만 지표 계산에서는 제외합니다.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            } }
            if (state.notes.isNotEmpty() && state.groups.isNotEmpty()) {
                item {
                Text("2D PCA 지도", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                PcaMap(state.projectedDocuments, state.notes, state.groups, onSelect = { selectedDocumentId = it.id })
                Text("PCA는 문서 임베딩을 2차원으로 줄여 보여주는 방법입니다. 점 간 거리는 원래 임베딩 거리와 달라질 수 있으므로 군집 품질은 정밀도·재현율·F1로 확인하세요.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            } else if (state.notes.isEmpty()) {
                item { EmptyCard("군집 실험 공간에 문서를 가져오면 지도가 표시됩니다.") }
            }
            }
        }
        if (state.workspace == "sample" && state.notes.isEmpty()) item { Button(onClick = viewModel::importSamples, enabled = !state.loading && state.modelReady) { Text("샘플 메모 가져오기") } }
        if (state.workspace == "experiment") {
            if (!state.groupCalculationActive) {
                if (state.notes.isNotEmpty() && state.groups.isEmpty()) item { EmptyCard("군집 결과가 없습니다. 위의 오류 안내를 확인하세요.") }
                else {
                    item {
                        Text(
                            "${grouped.size}개 군집 · 군집 문서 ${grouped.sumOf { it.value.notes.size }}개 · 미분류 문서 ${unclassified.size}개 · 전체 ${state.notes.size}개",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.SemiBold,
                        )
                    }
                    items(grouped.size, key = { grouped[it].value.id }) { rank ->
                        val (groupIndex, group) = grouped[rank]
                        ElevatedCard(Modifier.fillMaxWidth().clickable { selectedGroupId = group.id }) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Canvas(Modifier.size(12.dp)) { drawCircle(clusterColor(groupIndex, group)) }
                                    Text("군집 ${rank + 1}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                }
                                Text("${group.notes.size}개 문서", style = MaterialTheme.typography.titleSmall)
                                Text(group.notes.take(2).joinToString(" · ") { it.title }, style = MaterialTheme.typography.bodySmall, maxLines = 2)
                                Text("전체 문서 보기 ›", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                            }
                        }
                    }
                    if (unclassified.isNotEmpty()) item(key = "unclassified") {
                        ElevatedCard(Modifier.fillMaxWidth().clickable { unclassifiedOpen = true }) {
                            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    Canvas(Modifier.size(12.dp)) { drawCircle(Color(0xFF8A8A8A)) }
                                    Text("미분류", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                                }
                                Text("${unclassified.size}개 문서", style = MaterialTheme.typography.titleSmall)
                                Text(unclassified.take(2).joinToString(" · ") { it.title }, style = MaterialTheme.typography.bodySmall, maxLines = 2)
                                Text("전체 문서 보기 ›", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                            }
                        }
                    }
                }
            }
        } else {
            item { Text("${state.groups.size}개 묶음 · ${state.notes.size}개 메모", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold) }
            items(state.groups.size) { groupIndex ->
                val group = state.groups[groupIndex]
                ElevatedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(group.label, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text("${group.notes.size}개 메모", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        group.notes.take(5).forEach { note -> Text("• ${note.title}: ${note.text}", style = MaterialTheme.typography.bodySmall) }
                        if (group.notes.size > 5) Text("외 ${group.notes.size - 5}개", style = MaterialTheme.typography.labelSmall)
                        val groupKey = group.notes.map { it.id }.sorted().joinToString(",")
                        Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            TextButton(onClick = { viewModel.feedback("group", groupKey, true) }) { Text("👍 맞게 묶임") }
                            TextButton(onClick = { viewModel.feedback("group", groupKey, false) }) { Text("👎 잘못 묶임") }
                        }
                    }
                }
            }
        }
        item { Spacer(Modifier.height(12.dp)) }
    }
}

@Composable
private fun PcaMap(
    projected: List<ProjectedDocument>,
    notes: List<Note>,
    groups: List<NoteGroup>,
    onSelect: (Note) -> Unit,
) {
    val noteById = notes.associateBy(Note::id)
    val groupByNote = groups.flatMap { group -> group.notes.map { it.id to group.id } }.toMap()
    val groupColors = groups.mapIndexed { index, group -> group.id to clusterColor(index, group) }.toMap()
    val axisColor = MaterialTheme.colorScheme.outlineVariant
    val density = LocalDensity.current
    val margin = with(density) { 18.dp.toPx() }
    val hitRadius = with(density) { 30.dp.toPx() }
    var scale by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }
    Canvas(
        Modifier.fillMaxWidth().height(280.dp).pointerInput(projected) {
            detectTransformGestures { _, movement, zoom, _ ->
                scale = (scale * zoom).coerceIn(1f, 8f)
                pan += movement
            }
        }.pointerInput(projected, notes, groups, scale, pan) {
            detectTapGestures { tap ->
                val width = size.width.toFloat()
                val height = size.height.toFloat()
                val nearest = projected.mapNotNull { point ->
                    val note = noteById[point.noteId] ?: return@mapNotNull null
                    val rawX = margin + point.x.coerceIn(0f, 1f) * (width - margin * 2)
                    val rawY = height - margin - point.y.coerceIn(0f, 1f) * (height - margin * 2)
                    val x = (rawX - width / 2) * scale + width / 2 + pan.x
                    val y = (rawY - height / 2) * scale + height / 2 + pan.y
                    val dx = tap.x - x
                    val dy = tap.y - y
                    val distance = dx * dx + dy * dy
                    if (distance <= hitRadius * hitRadius) distance to note else null
                }.minByOrNull { it.first }
                nearest?.second?.let(onSelect)
            }
        },
    ) {
        val plotWidth = size.width - margin * 2
        val plotHeight = size.height - margin * 2
        drawLine(axisColor, Offset(margin, size.height - margin), Offset(size.width - margin, size.height - margin), strokeWidth = 1.dp.toPx())
        drawLine(axisColor, Offset(margin, margin), Offset(margin, size.height - margin), strokeWidth = 1.dp.toPx())
        val radius = if (projected.size > 400) 2.4.dp.toPx() else 4.dp.toPx()
        projected.forEach { point ->
            val color = groupColors[groupByNote[point.noteId]] ?: Color(0xFF8A8A8A)
            val rawX = margin + point.x.coerceIn(0f, 1f) * plotWidth
            val rawY = size.height - margin - point.y.coerceIn(0f, 1f) * plotHeight
            drawCircle(
                color = color,
                radius = radius * scale.coerceAtMost(2f),
                center = Offset((rawX - size.width / 2) * scale + size.width / 2 + pan.x, (rawY - size.height / 2) * scale + size.height / 2 + pan.y),
            )
        }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("PC1", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text("PC2", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        TextButton(onClick = { scale = 1f; pan = Offset.Zero }) { Text("지도 원위치") }
    }
    Text("점 ${projected.size}개 · 두 손가락으로 확대·이동 · 군집 색상은 아래 목록과 같습니다.", modifier = Modifier.padding(bottom = 8.dp), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

private val mapPalette = listOf(
    Color(0xFF386CB0), Color(0xFFF0027F), Color(0xFF7FC97F), Color(0xFFFF7F00),
    Color(0xFFBEAED4), Color(0xFF1B9E77), Color(0xFFD95F02), Color(0xFF7570B3),
)

private fun clusterColor(index: Int, group: NoteGroup) =
    if (group.notes.size == 1) Color(0xFF8A8A8A) else mapPalette[index % mapPalette.size]

@Composable
private fun DocumentImportPreviewDialog(
    previews: List<ImportPreviewItem>,
    enabled: Boolean,
    onLabelChange: (String, String) -> Unit,
    onDismiss: () -> Unit,
    onConfirm: () -> Unit,
) {
    val importableCount = previews.count { it.document != null }
    AlertDialog(
        onDismissRequest = { if (enabled) onDismiss() },
        title = { Text("문서 확인 및 정답 주제 지정") },
        text = {
            Column(
                Modifier.fillMaxWidth().height(460.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("정답 주제를 같은 이름으로 입력한 문서는 하나의 정답 군집으로 평가합니다. 비워 두면 문서는 군집에 포함되지만 품질 지표에서는 제외합니다.", style = MaterialTheme.typography.bodySmall)
                previews.forEach { item ->
                    ElevatedCard(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                            Text(item.fileName, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                            if (item.error != null) {
                                Text(item.error, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                            } else {
                                val document = item.document!!
                                Text(
                                    if (document.pageTexts.isEmpty()) "${document.text.length}자"
                                    else "PDF · ${document.pageTexts.size}쪽 · ${document.pageTexts.sumOf(String::length)}자 추출",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                                if (document.emptyPageNumbers.isNotEmpty()) {
                                    Text("글을 추출하지 못한 쪽: ${document.emptyPageNumbers.joinToString(", ")}", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                                }
                                OutlinedTextField(
                                    value = item.groundTruthGroup,
                                    onValueChange = { onLabelChange(document.id, it) },
                                    modifier = Modifier.fillMaxWidth(),
                                    label = { Text("정답 주제 (선택)") },
                                    singleLine = true,
                                )
                                Text(document.text.take(1200) + if (document.text.length > 1200) "…" else "", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onConfirm, enabled = enabled && importableCount > 0) { Text("${importableCount}개 가져오기") }
        },
        dismissButton = { TextButton(onClick = onDismiss, enabled = enabled) { Text("취소") } },
    )
}

@Composable
private fun GroundTruthLabelDialog(note: Note, onDismiss: () -> Unit, onSave: (String) -> Unit) {
    var label by remember(note.id) { mutableStateOf(note.groundTruthGroup.orEmpty()) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("정답 주제 수정") },
        text = {
            OutlinedTextField(
                value = label,
                onValueChange = { label = it },
                label = { Text("정답 주제") },
                placeholder = { Text("비우면 품질 지표에서 제외") },
                singleLine = true,
            )
        },
        confirmButton = { TextButton(onClick = { onSave(label) }) { Text("저장") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("취소") } },
    )
}

@Composable
private fun SearchScreen(viewModel: PocViewModel) {
    val state = viewModel.state
    Column(Modifier.fillMaxSize().padding(16.dp)) {
        Text("기기 안의 메모만 검색합니다. 검색어는 측정 보고서에 저장하지 않습니다.", style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(10.dp))
        OutlinedTextField(
            value = state.query,
            onValueChange = viewModel::setQuery,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("검색어") },
            placeholder = { Text("예: 아이 병원 어디로 가기로 했었지?") },
            singleLine = true,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.padding(vertical = 8.dp)) {
            SearchMode.values().forEach { mode ->
                FilterChip(selected = state.searchMode == mode, onClick = { viewModel.setSearchMode(mode) }, label = { Text(mode.label) })
            }
        }
        Button(onClick = viewModel::search, enabled = state.modelReady && !state.loading) { Text("검색") }
        if (state.notes.isEmpty()) Text("현재 공간에 검색할 메모가 없습니다.", Modifier.padding(top = 16.dp), style = MaterialTheme.typography.bodyMedium)
        else if (state.hits.isEmpty()) Text("검색어를 입력하고 검색을 누르세요.", Modifier.padding(top = 16.dp), style = MaterialTheme.typography.bodyMedium)
        else LazyColumn(contentPadding = PaddingValues(top = 12.dp, bottom = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(state.hits, key = { it.note.id }) { hit ->
                ElevatedCard(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                        Text(hit.note.title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                        Text(hit.note.text, style = MaterialTheme.typography.bodySmall)
                        Text("점수 ${"%.4f".format(Locale.ROOT, hit.score)}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        Row {
                            TextButton(onClick = { viewModel.feedback("search", hit.note.id, true) }) { Text("👍 관련 있음") }
                            TextButton(onClick = { viewModel.feedback("search", hit.note.id, false) }) { Text("👎 관련 없음") }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun EvaluationScreen(viewModel: PocViewModel, onExport: () -> Unit) {
    val state = viewModel.state
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        ElevatedCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("고정 평가 자료", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text("1,000개 테스트 메모 · 140개 검색 질문 · 100개 문서 쌍", style = MaterialTheme.typography.bodySmall)
                Text("합성 자료의 자동 정답은 사람 검수 전 잠정치입니다. 실제 기기 성능과 7일 사용 결과는 측정 전 미판정으로 둡니다.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                Button(onClick = viewModel::runEvaluation, enabled = state.modelReady && !state.loading) { Text("기기에서 고정 자료 평가") }
            }
        }
        state.evaluationProgress?.let { (current, total) ->
            LinearProgressIndicator(progress = { current.toFloat() / total }, modifier = Modifier.fillMaxWidth())
            Text("임베딩 $current/$total")
        }
        state.evaluation?.let { result ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("검색 · 잠정", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    SearchMode.values().forEach { mode ->
                        val metrics = result.retrieval.getValue(mode)
                        Text("${mode.label}: Recall@1 ${percent(metrics.recallAt1)} · Recall@5 ${percent(metrics.recallAt5)} · MRR ${percent(metrics.mrr)}")
                    }
                }
            }
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("묶음 · threshold ${result.threshold}", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text("표본 쌍 F1 ${percent(result.sampledPairs.f1)} · 정밀도 ${percent(result.sampledPairs.precision)} · 재현율 ${percent(result.sampledPairs.recall)} (${result.pairCount}쌍)")
                    Text("전체 사건 쌍 F1 ${percent(result.allEventPairs.f1)} · 정밀도 ${percent(result.allEventPairs.precision)} · 재현율 ${percent(result.allEventPairs.recall)}")
                    Text("과도하게 합친 문서 쌍 ${result.allEventPairs.falseMerges}", color = if (result.allEventPairs.falseMerges > 0) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary)
                    Text("데이터 revision ${result.datasetRevision}", style = MaterialTheme.typography.labelSmall)
                }
            }
            if (result.mixedGroups.isNotEmpty()) {
                Text("사건이 섞여 잘못 합쳐진 묶음", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                result.mixedGroups.forEach { mixed ->
                    ElevatedCard(Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(5.dp)) {
                            Text("${mixed.group.notes.size}개 메모 · ${mixed.eventCount}개 사건", color = MaterialTheme.colorScheme.error, fontWeight = FontWeight.SemiBold)
                            mixed.group.notes.take(4).forEach { Text("• ${it.title}: ${it.text}", style = MaterialTheme.typography.bodySmall) }
                            val key = mixed.group.notes.map { it.id }.sorted().joinToString(",")
                            Row {
                                TextButton(onClick = { viewModel.feedback("evaluation_group", key, true) }) { Text("👍 맞음") }
                                TextButton(onClick = { viewModel.feedback("evaluation_group", key, false) }) { Text("👎 잘못 묶임") }
                            }
                        }
                    }
                }
            }
        }
        ElevatedCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("실제 기기 성능", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                state.device?.let { device ->
                    Text("${device.manufacturer} ${device.model} · Android ${device.androidRelease} (API ${device.sdk}) · ${device.abi ?: "ABI 미확인"}", style = MaterialTheme.typography.bodySmall)
                    Text("RAM ${"%.1f".format(Locale.ROOT, device.ramBytes / 1_000_000_000.0)}GB · 사용 가능 CPU 코어 ${device.cpuCores} · ${if (device.meetsPocTarget) "POC 기준 기기" else "POC 기준 미충족"}", style = MaterialTheme.typography.bodySmall, color = if (device.meetsPocTarget) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
                }
                Text("콜드 30회 · 재사용 100회 · 검색 한 번 2초 상한 · 최대 추가 PSS 1.5GB 기준", style = MaterialTheme.typography.bodySmall)
                Button(onClick = viewModel::runPerformanceMeasurement, enabled = state.modelReady && !state.loading) { Text("성능 측정 시작") }
                state.performance?.let { value ->
                    Text("콜드 검색 p95 ${formatMs(value.coldP95Ms)} · 재사용 p95 ${formatMs(value.warmP95Ms)}")
                    Text("실패 ${value.coldFailures + value.warmFailures}회 · 최대 추가 PSS ${"%.0f".format(value.maxAdditionalPssMb)}MB · 모델 파일 ${"%.1f".format(value.modelFilesBytes / 1_000_000_000.0)}GB")
                    Text("측정 시각 ${formatDate(value.measuredAt)} · 기준 기기 여부는 기기 정보로 확인하세요.", style = MaterialTheme.typography.labelSmall)
                } ?: Text("기기 측정 전 · 미판정", color = MaterialTheme.colorScheme.error)
            }
        }
        ElevatedCard(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text("최근 7일 사용 요약", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text("메모와 검색어는 저장하지 않고 날짜별 👍/👎 수와 마지막 지연 측정만 집계합니다.", style = MaterialTheme.typography.bodySmall)
                if (state.dailySummaries.isEmpty()) Text("아직 기록이 없습니다. 검색 결과와 묶음에 의견을 남기고 성능을 측정해 보세요.")
                state.dailySummaries.forEach { day ->
                    Text("${day.day} · 👍 ${day.positive} / 👎 ${day.negative} · 콜드 ${day.coldP95Ms?.let(::formatMs) ?: "미측정"} · 재사용 ${day.warmP95Ms?.let(::formatMs) ?: "미측정"} · 실패 ${day.failedRuns}", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        OutlinedButton(onClick = onExport, enabled = state.modelReady) { Text("측정 보고서 내보내기") }
        Text("보고서는 기기·모델·데이터 revision, 해시, 평가와 성능 수치만 포함합니다. 개인 메모와 검색어는 포함하지 않습니다.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun NoteEditorDialog(note: Note?, onDismiss: () -> Unit, onSave: (String?, String, String) -> Unit) {
    var title by remember(note?.id) { mutableStateOf(note?.title.orEmpty()) }
    var text by remember(note?.id) { mutableStateOf(note?.text.orEmpty()) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(if (note == null) "메모 작성" else "메모 수정") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(title, { title = it }, label = { Text("제목") }, singleLine = true)
                OutlinedTextField(text, { text = it }, label = { Text("내용") }, minLines = 5)
            }
        },
        confirmButton = { TextButton(onClick = { onSave(note?.id, title, text) }) { Text("저장") } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("취소") } },
    )
}

@Composable
private fun EmptyCard(message: String) {
    Surface(Modifier.fillMaxWidth(), color = MaterialTheme.colorScheme.surfaceContainer, shape = MaterialTheme.shapes.medium) {
        Text(message, Modifier.padding(16.dp), style = MaterialTheme.typography.bodyMedium)
    }
}

private fun percent(value: Double) = "%.1f%%".format(Locale.ROOT, value * 100)
private fun formatMs(value: Double) = if (value.isFinite()) "%.0fms".format(Locale.ROOT, value) else "시간 초과"
private fun formatDate(value: Long) = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.KOREA).format(Date(value))
