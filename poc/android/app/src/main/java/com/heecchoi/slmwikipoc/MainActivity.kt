package com.heecchoi.slmwikipoc

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
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
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
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
    var editing by remember { mutableStateOf<Note?>(null) }
    var showEditor by remember { mutableStateOf(false) }
    var deleting by remember { mutableStateOf<Note?>(null) }
    val reportLauncher = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { viewModel.exportReport(it) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text("메모 검색 · 묶음 POC", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(state.modelStatus, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                },
            )
        },
        bottomBar = {
            NavigationBar {
                AppTab.values().forEach { tab ->
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
                AppTab.NOTES -> NotesScreen(viewModel, onEdit = { editing = it; showEditor = true }, onDelete = { deleting = it })
                AppTab.GROUPS -> GroupsScreen(viewModel)
                AppTab.SEARCH -> SearchScreen(viewModel)
                AppTab.EVALUATION -> EvaluationScreen(viewModel) { reportLauncher.launch("embedding-poc-report.json") }
            }
        }
    }

    if (showEditor) NoteEditorDialog(editing, onDismiss = { showEditor = false }, onSave = { id, title, text -> viewModel.saveNote(id, title, text); showEditor = false })
    deleting?.let { note ->
        AlertDialog(
            onDismissRequest = { deleting = null },
            title = { Text("메모를 삭제할까요?") },
            text = { Text("‘${note.title}’ 메모와 이 메모의 로컬 임베딩을 삭제합니다.") },
            confirmButton = { TextButton(onClick = { viewModel.deleteNote(note); deleting = null }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } },
        )
    }
}

@Composable
private fun NotesScreen(viewModel: PocViewModel, onEdit: (Note) -> Unit, onDelete: (Note) -> Unit) {
    val state = viewModel.state
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("개인 메모는 이 기기의 앱 전용 저장소에만 저장합니다.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = state.workspace == "personal", onClick = { viewModel.selectWorkspace("personal") }, label = { Text("개인") })
            FilterChip(selected = state.workspace == "sample", onClick = { viewModel.selectWorkspace("sample") }, label = { Text("합성 샘플") })
        }
        if (state.workspace == "sample") {
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("개인 메모와 분리된 합성 자료입니다.", fontWeight = FontWeight.SemiBold)
                    Text("고정 평가 데이터와 별도 공간에 저장합니다.", style = MaterialTheme.typography.bodySmall)
                    Button(onClick = viewModel::importSamples, enabled = !state.loading && state.modelReady) { Text("샘플 메모 가져오기") }
                }
            }
        }
        if (state.notes.isEmpty()) {
            EmptyCard(if (state.workspace == "personal") "아직 메모가 없습니다. 아래 버튼으로 메모를 작성해 보세요." else "샘플이 없습니다. 위에서 합성 샘플을 가져오세요.")
        } else {
            Text("${state.notes.size}개 메모", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
            state.notes.forEach { note ->
                ElevatedCard(Modifier.fillMaxWidth().clickable { if (note.workspace == "personal") onEdit(note) }) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Text(note.title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(note.text, style = MaterialTheme.typography.bodyMedium)
                        Text(formatDate(note.updatedAt), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (note.workspace == "personal") Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = { onEdit(note) }) { Text("수정") }
                            TextButton(onClick = { onDelete(note) }) { Text("삭제", color = MaterialTheme.colorScheme.error) }
                        }
                    }
                }
            }
        }
        Spacer(Modifier.height(72.dp))
    }
}

@Composable
private fun GroupsScreen(viewModel: PocViewModel) {
    val state = viewModel.state
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("${if (state.workspace == "personal") "개인 메모" else "합성 샘플"}를 코사인 거리 평균 연결 방식으로 묶습니다. 개발 자료에서 고른 거리 기준은 ${"%.2f".format(Locale.ROOT, state.clusterThreshold)}입니다.", style = MaterialTheme.typography.bodySmall)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilterChip(selected = state.workspace == "personal", onClick = { viewModel.selectWorkspace("personal") }, label = { Text("개인") })
            FilterChip(selected = state.workspace == "sample", onClick = { viewModel.selectWorkspace("sample") }, label = { Text("합성 샘플") })
        }
        if (state.workspace == "sample" && state.notes.isEmpty()) Button(onClick = viewModel::importSamples, enabled = !state.loading && state.modelReady) { Text("샘플 메모 가져오기") }
        Text("${state.groups.size}개 묶음 · ${state.notes.size}개 메모", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        state.groups.forEach { group ->
            ElevatedCard(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(group.label, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text("${group.notes.size}개 메모", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    group.notes.take(5).forEach { Text("• ${it.title}: ${it.text}", style = MaterialTheme.typography.bodySmall) }
                    if (group.notes.size > 5) Text("외 ${group.notes.size - 5}개", style = MaterialTheme.typography.labelSmall)
                    val groupKey = group.notes.map { it.id }.sorted().joinToString(",")
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        TextButton(onClick = { viewModel.feedback("group", groupKey, true) }) { Text("👍 맞게 묶임") }
                        TextButton(onClick = { viewModel.feedback("group", groupKey, false) }) { Text("👎 잘못 묶임") }
                    }
                }
            }
        }
        Spacer(Modifier.height(12.dp))
    }
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
