package com.heecchoi.slmwikipoc

import android.content.ContentValues
import android.content.Context
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class LifeRepositoryTest {
    private lateinit var context: Context
    private lateinit var dbName: String
    private lateinit var preferences: String
    private lateinit var store: LocalStore
    private lateinit var repo: LifeRepository

    @Before fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        val suffix = UUID.randomUUID().toString()
        dbName = "life-test-" + suffix + ".db"
        preferences = "life-test-" + suffix
        store = LocalStore(context, dbName)
        repo = LifeRepository(context, store, preferences) { }
    }

    @After fun cleanup() {
        store.close()
        context.deleteDatabase(dbName)
        context.getSharedPreferences(preferences, Context.MODE_PRIVATE).edit().clear().commit()
    }

    @Test fun allowedNotificationsKeepDistinctUpdatesAndExpireWithoutDeletingApprovedTask() {
        val now = System.currentTimeMillis()
        val old = now - LifeRepository.RETENTION_MS + 3_600_000L
        repo.setAllowed("example.shop", true)
        repo.setCollectionEnabled(true)
        assertFalse(repo.recordNotification("other.app", "다른 앱", "other", "비공개", "저장되지 않음", now, "Asia/Seoul", now))
        assertTrue(repo.recordNotification("example.shop", "상점", "key-1", "배송", "내일 배송 예정", old, "Asia/Seoul", old))
        assertFalse(repo.recordNotification("example.shop", "상점", "key-1", "배송", "내일 배송 예정", old, "Asia/Seoul", old + 1))
        val first = repo.records().single()
        val candidate = repo.candidates().single()
        repo.approveCandidate(candidate, "event", "택배 도착", "2026-11-06")
        assertTrue(repo.recordNotification("example.shop", "상점", "key-1", "배송", "배송 완료", now, "Asia/Seoul", now))
        assertEquals(2, repo.versions(first.id).size)
        repo.cleanupExpired(now + 3_600_001L)
        assertEquals(1, repo.versions(first.id, now + 3_600_001L).size)
        val task = repo.tasks().single()
        assertEquals("택배 도착", task.title)
        assertTrue(task.sourceExpired)
        assertEquals("배송 완료", repo.records(now + 3_600_001L).single().text.trim().lines().last())
    }

    @Test fun deletionPreventsSameActiveNotificationReturningAndDisablingKeepsOldRecords() {
        val now = System.currentTimeMillis()
        repo.setAllowed("example.shop", true)
        repo.setCollectionEnabled(true)
        assertTrue(repo.recordNotification("example.shop", "상점", "key-2", "주문", "주문번호: A1234", now, "UTC", now))
        val id = repo.records().single().id
        repo.setCollectionEnabled(false)
        assertEquals(1, repo.records().size)
        assertFalse(repo.recordNotification("example.shop", "상점", "key-2", "주문", "새 내용", now, "UTC", now + 1))
        repo.setCollectionEnabled(true)
        repo.deleteRecord(id)
        assertFalse(repo.recordNotification("example.shop", "상점", "key-2", "주문", "주문번호: A1234", now, "UTC", now + 2))
        assertTrue(repo.records().isEmpty())
        repo.notificationRemoved("key-2", false)
        assertTrue(repo.recordNotification("example.shop", "상점", "key-2", "주문", "주문번호: A1234", now, "UTC", now + 3))
    }

    @Test fun notificationLockdownRemovesSourceAndKeepsConfirmedTask() {
        val now = System.currentTimeMillis()
        repo.setAllowed("example.shop", true)
        repo.setCollectionEnabled(true)
        repo.recordNotification("example.shop", "상점", "key-3", "반품", "반품 신청", now, "UTC", now)
        val candidate = repo.candidates().first()
        repo.approveCandidate(candidate, "task", "반품 신청", null)
        repo.notificationRemoved("key-3", true)
        assertTrue(repo.records().isEmpty())
        assertTrue(repo.tasks().single().sourceExpired)
    }

    @Test fun manualGroupsAndExtraTopicsSurviveEditsAndAutomaticComparisons() {
        val first = repo.addText("배송 안내", "배송 예정", false)
        val second = repo.addText("주문 기록", "주문 내역", false)
        val topic = repo.createGroup("topic", "쇼핑", listOf(first, second))
        val extra = repo.createGroup("topic", "여행", emptyList())
        repo.addMember(extra, first, extra = true)
        val order = repo.createGroup("order", "선물 주문", listOf(first, second))
        repo.setTruthOrder(first, "정답 주문")

        repo.saveRun(0.05, "data", "model", listOf(listOf(first, second)), listOf(listOf(first), listOf(second)))
        repo.updateText(first, "수정한 배송 안내", "배송 도착")
        assertEquals(setOf(first to "primary", second to "primary"), repo.manualGroups("topic").single { it.id == topic }.members.toSet())
        assertEquals(listOf(first to "extra"), repo.manualGroups("topic").single { it.id == extra }.members)
        assertEquals(setOf(first, second), repo.manualGroups("order").single { it.id == order }.members.map { it.first }.toSet())
        assertEquals("정답 주문", repo.records().single { it.id == first }.truthOrder)

        val moved = repo.createGroup("topic", "다른 주제", listOf(first))
        assertEquals(listOf(second to "primary"), repo.manualGroups("topic").single { it.id == topic }.members)
        assertEquals(listOf(first to "primary"), repo.manualGroups("topic").single { it.id == moved }.members)
        assertEquals(listOf(first to "extra"), repo.manualGroups("topic").single { it.id == extra }.members)
    }

    @Test fun migrationPreservesExistingDocumentAndReferenceAcrossUpdate() {
        store.close()
        context.deleteDatabase(dbName)
        val db = context.openOrCreateDatabase(dbName, Context.MODE_PRIVATE, null)
        db.execSQL("""CREATE TABLE notes (
            id TEXT PRIMARY KEY NOT NULL, workspace TEXT NOT NULL, event_id TEXT, title TEXT NOT NULL,
            body TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, search_vectors BLOB NOT NULL,
            cluster_vector BLOB NOT NULL, ground_truth_group TEXT, source_name TEXT, source_mime_type TEXT,
            source_path TEXT, page_texts TEXT, device_path TEXT, device_size INTEGER, device_modified_at INTEGER)""")
        val vector = FloatArray(768) { if (it == 0) 1f else 0f }
        db.insertOrThrow("notes", null, ContentValues().apply {
            put("id", "old"); put("workspace", "experiment"); put("title", "기존 문서")
            put("body", "원래 글"); put("created_at", 1L); put("updated_at", 1L)
            put("search_vectors", FloatCodec.encode(listOf(vector))); put("cluster_vector", FloatCodec.encode(listOf(vector)))
        })
        db.version = 3
        db.close()
        store = LocalStore(context, dbName)
        repo = LifeRepository(context, store, preferences) { }
        assertEquals("원래 글", store.note("old")?.text)
        assertTrue(repo.linkDocument("old"))
        val linkedId = repo.records().single().id
        val group = repo.createGroup("topic", "여행", listOf(linkedId))
        store.upsert(NoteInput("old", null, "기존 문서", "바뀐 글"), listOf(vector), vector, "experiment")
        assertEquals("바뀐 글", repo.records().single().text)
        assertEquals(linkedId, repo.manualGroups("topic").single { it.id == group }.members.single().first)
        store.delete("old")
        assertTrue(repo.records().isEmpty())
        assertNotNull(repo.manualGroups("topic").single { it.id == group })
    }
}
