package com.heecchoi.slmwikipoc

import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class ExperimentStoreTest {
    private lateinit var context: Context
    private lateinit var dbName: String
    private lateinit var prefs: String
    private lateinit var local: LocalStore
    private lateinit var repo: ExperimentStore
    private val vector = FloatArray(768) { if (it == 0) 1f else 0f }

    @Before fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        val suffix = UUID.randomUUID().toString()
        dbName = "experiment-test-$suffix.db"
        prefs = "experiment-test-$suffix"
        local = LocalStore(context, dbName)
        repo = ExperimentStore(context, local, prefs)
    }

    @After fun cleanup() {
        local.close()
        context.deleteDatabase(dbName)
        context.getSharedPreferences(prefs, Context.MODE_PRIVATE).edit().clear().commit()
    }

    private fun doc(text: String = "문서 본문") = ParsedDocument("doc", "문서.txt", "text/plain", "문서", text,
        emptyList(), emptyList(), null)

    @Test fun upgradeKeepsLegacyDataAndStartsNewExperimentEmpty() {
        val path = "/storage/emulated/0/legacy.txt"
        val source = DocumentSource("legacy.txt", "text/plain", devicePath = path, deviceSize = 10L, deviceModifiedAt = 20L)
        local.upsert(NoteInput("legacy-note", null, "이전 문서", "이전 본문", "기존 정답", source), listOf(vector), vector, "experiment")
        val life = LifeRepository(context, local, prefs + "-life") { }
        assertTrue(life.linkDocument("legacy-note"))
        val lifeId = life.records().single().id
        val group = life.createGroup("topic", "기존 주제", listOf(lifeId))
        val db = local.writableDatabase
        db.execSQL("DROP TABLE experiment_run_items")
        db.execSQL("DROP TABLE experiment_run")
        db.execSQL("DROP TABLE experiment_inputs")
        db.version = 4
        local.close()
        local = LocalStore(context, dbName)
        repo = ExperimentStore(context, local, prefs)
        assertTrue(repo.items().isEmpty())
        assertEquals("이전 본문", local.note("legacy-note")?.text)
        val migratedLife = LifeRepository(context, local, prefs + "-life") { }
        assertEquals(lifeId, migratedLife.manualGroups("topic").single { it.id == group }.members.single().first)
        assertTrue(repo.saveScanned(doc("새로 읽은 본문"), source, local.note("legacy-note")?.groundTruthGroup))
        val fresh = repo.items().single()
        assertEquals("기존 정답", fresh.label)
        assertNull(fresh.vector)
        assertEquals("이전 본문", local.note("legacy-note")?.text)
        assertEquals(lifeId, migratedLife.records().single().id)
        context.getSharedPreferences(prefs + "-life", Context.MODE_PRIVATE).edit().clear().commit()
    }

    @Test fun runAndInputDataSurviveRestartAndDeletionRemovesOnlyItsPoint() {
        repo.saveText(null, "text", "", "직접 입력 본문")
        repo.saveText(null, "voice", "음성 제목", "인식된 음성")
        var items = repo.items()
        assertEquals(2, items.size)
        assertTrue(items.all { it.vector == null })
        val first = items.single { it.kind == "text" }
        val second = items.single { it.kind == "voice" }
        assertEquals("직접 입력 본문", first.title)
        assertTrue(repo.cacheVector(first, vector, "model:tokenizer"))
        assertTrue(repo.cacheVector(second, vector, "model:tokenizer"))
        items = repo.items()
        val notes = items.map { it.note() }
        val groups = listOf(NoteGroup(0, "", notes))
        val projection = notes.mapIndexed { index, note -> ProjectedDocument(note.id, index / 2f, index / 2f) }
        val signature = ExperimentStore.signature(items, repo.threshold(), "model:tokenizer")
        val quality = ExperimentQualityCalculator.calculate(items, groups)
        assertTrue(repo.saveRun(signature, "model:tokenizer", repo.threshold(), groups, projection, quality, 100, 7, 2, 0))
        assertEquals(2, repo.lastRun()!!.points.size)
        var checks = 0
        val cancelled = runCatching {
            repo.saveRun(signature, "model:tokenizer", repo.threshold(), groups, projection, quality, 200, 8, 2, 0) {
                if (++checks == 3) error("취소")
            }
        }
        assertTrue(cancelled.isFailure)
        assertEquals(100L, repo.lastRun()!!.embeddingMs)
        assertEquals(2, repo.lastRun()!!.points.size)
        repo.remove(second.id)
        val oldResult = repo.lastRun()!!
        assertEquals(listOf(first.id), oldResult.points.map(ExperimentRunPoint::inputId))
        assertEquals(2, oldResult.points.single().groupSize)
        assertFalse(oldResult.signature == ExperimentStore.signature(repo.items(), repo.threshold(), "model:tokenizer"))
        local.close()
        local = LocalStore(context, dbName)
        repo = ExperimentStore(context, local, prefs)
        assertEquals(first.id, repo.items().single().id)
        assertEquals(first.id, repo.lastRun()!!.points.single().inputId)
        repo.saveText(first.id, "text", "수정 제목", "새 본문")
        assertFalse(repo.cacheVector(first, vector, "model:tokenizer"))
        assertNull(repo.items().single().vector)
        assertFalse(repo.saveRun(signature, "model:tokenizer", repo.threshold(), groups, projection, quality, 100, 7, 2, 0))
        assertNotNull(repo.lastRun())
        repo.clear()
        assertTrue(repo.items().isEmpty())
        assertNull(repo.lastRun())
    }

    @Test fun repeatedScanUpdatesOneRecordAndNeverDeletesOriginalFile() {
        val file = File(context.cacheDir, "scan-source-${UUID.randomUUID()}.txt")
        file.writeText("원본 파일")
        try {
            val source = DocumentSource("문서.txt", "text/plain", devicePath = file.canonicalPath,
                deviceSize = file.length(), deviceModifiedAt = file.lastModified())
            assertTrue(repo.saveScanned(doc(), source, "여행"))
            val first = repo.items().single()
            assertTrue(repo.cacheVector(first, vector, "model:tokenizer"))
            repo.markScanError(first.id, "읽기 실패")
            assertNotNull(repo.items().single().scanError)
            assertFalse(repo.saveScanned(doc("변경한 본문"), source, null))
            val updated = repo.items().single()
            assertEquals(first.id, updated.id)
            assertEquals("여행", updated.label)
            assertEquals("변경한 본문", updated.body)
            assertNull(updated.vector)
            assertNull(updated.scanError)
            repo.remove(updated.id)
            assertTrue(file.exists())
            assertEquals("원본 파일", file.readText())
        } finally { file.delete() }
    }

    @Test fun cancelledMissingFileCleanupDoesNotRemovePartialSet() {
        val sourceA = DocumentSource("a.txt", "text/plain", devicePath = "/storage/a.txt", deviceSize = 1, deviceModifiedAt = 1)
        val sourceB = DocumentSource("b.txt", "text/plain", devicePath = "/storage/b.txt", deviceSize = 1, deviceModifiedAt = 1)
        repo.saveScanned(doc("a"), sourceA, null)
        repo.saveScanned(doc("b"), sourceB, null)
        val ids = repo.scanned().map(ScannedInputRecord::id)
        var checks = 0
        assertTrue(runCatching { repo.removeMissingScanned(ids) { if (++checks == 2) error("취소") } }.isFailure)
        assertEquals(2, repo.items().size)
        assertEquals(2, repo.removeMissingScanned(ids))
        assertTrue(repo.items().isEmpty())
    }

    @Test fun documentTypedAndVoiceInputsUseOneExperimentAndPreserveOriginal() {
        val file = File(context.cacheDir, "input-${UUID.randomUUID()}.txt")
        file.writeText("여행 예약 원문", Charsets.UTF_8)
        try {
            val document = DocumentImportParser.parseFile(context, file)
            val source = DocumentImportParser.persistSource(document, File(context.cacheDir, "experiment-test-pdfs"))
            repo.addDocument(document, source, "여행")
            repo.saveText(null, "text", "", "여행 메모")
            repo.saveText(null, "voice", "", "여행 음성 인식 글")
            val before = repo.items()
            assertEquals(setOf("document", "text", "voice"), before.map(ExperimentItem::kind).toSet())
            assertTrue(before.all { it.vector == null })
            before.filter { it.kind != "document" }.forEach { repo.setLabel(it.id, "여행") }
            val labeled = repo.items()
            labeled.forEach { assertTrue(repo.cacheVector(it, vector, "model:tokenizer")) }
            val cached = repo.items()
            val notes = cached.map { it.note() }
            val groups = AverageLinkageGroups.clusterLarge(notes, 0.05)
            assertEquals(cached.map(ExperimentItem::id).toSet(), groups.flatMap { it.notes.map(Note::id) }.toSet())
            assertEquals(1.0, ExperimentQualityCalculator.calculate(cached, groups).f1!!, 0.0)
            repo.addDocument(document, source, null)
            assertEquals(4, repo.items().size)
            repo.clear()
            assertEquals("여행 예약 원문", file.readText(Charsets.UTF_8))
        } finally { file.delete() }
    }
}
