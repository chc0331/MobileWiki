package com.heecchoi.slmwikipoc

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.update
import org.json.JSONArray
import org.json.JSONObject
import java.security.MessageDigest
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.util.UUID
import kotlin.math.sqrt

object LifeSignals {
    val revision = MutableStateFlow(0L)
    val connected = MutableStateFlow(false)
    fun changed() { revision.update { it + 1 } }
}

class LifeRepository(
    context: Context,
    private val store: LocalStore = LifeServices.store(context),
    preferencesName: String = "life-poc",
    private val schedule: (Context) -> Unit = { LifeWork.enqueue(it) },
) {
    private val app = context.applicationContext
    private val prefs = app.getSharedPreferences(preferencesName, Context.MODE_PRIVATE)

    companion object {
        const val RETENTION_MS = 30L * 24 * 60 * 60 * 1000
        const val MAX_TEXT = 20_000
        fun digest(text: String): String = MessageDigest.getInstance("SHA-256")
            .digest(text.toByteArray(Charsets.UTF_8)).joinToString("") { "%02x".format(it) }
    }

    @Synchronized fun allowedPackages(): Set<String> = prefs.getStringSet("allowed", emptySet())!!.toSet()
    @Synchronized fun collectionEnabled(): Boolean = prefs.getBoolean("enabled", false)
    @Synchronized fun acceptsPackage(packageName: String): Boolean =
        collectionEnabled() && packageName != app.packageName && packageName in allowedPackages()

    @Synchronized fun setAllowed(packageName: String, allowed: Boolean) {
        val packages = allowedPackages().toMutableSet()
        if (allowed) packages.add(packageName) else packages.remove(packageName)
        prefs.edit().putStringSet("allowed", packages).commit()
        LifeSignals.changed()
    }

    @Synchronized fun setCollectionEnabled(enabled: Boolean) {
        prefs.edit().putBoolean("enabled", enabled).commit()
        LifeSignals.changed()
    }

    fun currentModelHash(): String = JSONObject(app.assets.open("model/embedding_encoder.manifest.json")
        .bufferedReader().use { it.readText() }).getString("onnx_sha256")

    private fun calibrationValid(): Boolean = prefs.getString("calibration_model", null) == currentModelHash() &&
        prefs.getString("calibration_dataset", null) == LifeSynthetic.DEV &&
        reviewedIds(LifeSynthetic.DEV).containsAll(LifeSynthetic.orders(LifeSynthetic.DEV).keys)

    fun threshold(): Double? = if (prefs.contains("threshold") && calibrationValid()) prefs.getFloat("threshold", 0.05f).toDouble() else null
    fun topicThreshold(): Double? = if (prefs.contains("topic_threshold") && calibrationValid()) prefs.getFloat("topic_threshold", 0.05f).toDouble() else null
    fun setThreshold(value: Double, topic: Double, modelHash: String) {
        check(!trialActive()) { "7일 시험 중에는 기준값을 변경할 수 없습니다." }
        check(modelHash == currentModelHash()) { "기준값을 정한 모델이 현재 모델과 다릅니다." }
        check(reviewedIds(LifeSynthetic.DEV).containsAll(LifeSynthetic.orders(LifeSynthetic.DEV).keys)) { "개발 자료 검토를 먼저 마쳐 주세요." }
        prefs.edit().putFloat("threshold", value.toFloat()).putFloat("topic_threshold", topic.toFloat())
            .putString("calibration_model", modelHash).putString("calibration_dataset", LifeSynthetic.DEV).commit()
        LifeSignals.changed()
    }

    fun addText(title: String, body: String, shared: Boolean): String {
        val cleanTitle = title.trim()
        val cleanBody = body.trim()
        require(cleanTitle.isNotEmpty() && cleanBody.isNotEmpty()) { "제목과 내용을 입력해 주세요." }
        require(cleanBody.length <= MAX_TEXT) { "메모와 공유 글은 20,000자까지 저장할 수 있습니다." }
        val now = System.currentTimeMillis()
        val id = UUID.randomUUID().toString()
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            db.insertOrThrow("life_records", null, ContentValues().apply {
                put("id", id); put("kind", if (shared) "share" else "memo")
                put("title", cleanTitle); put("body", cleanBody)
                put("created_at", now); put("updated_at", now)
                if (!shared) put("source_time", now)
                put("zone_id", ZoneId.systemDefault().id)
            })
            val base = if (shared) null else Instant.ofEpochMilli(now).atZone(ZoneId.systemDefault()).toLocalDate()
            addCandidates(db, id, null, cleanTitle + "\n" + cleanBody, base)
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
        schedule(app)
        return id
    }

    fun linkDocument(noteId: String): Boolean {
        val note = store.note(noteId) ?: return false
        val now = System.currentTimeMillis()
        val id = "doc-" + UUID.randomUUID()
        val db = store.writableDatabase
        db.beginTransaction()
        var inserted = false
        try {
            inserted = db.insertWithOnConflict("life_records", null, ContentValues().apply {
                put("id", id); put("kind", "document"); put("note_id", noteId); put("note_updated_at", note.updatedAt)
                put("created_at", now); put("updated_at", now); put("processing", "ready")
            }, SQLiteDatabase.CONFLICT_IGNORE) != -1L
            if (inserted) addCandidates(db, id, null, note.title + "\n" + note.text, null)
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        if (inserted) LifeSignals.changed()
        return inserted
    }

    @Synchronized fun recordNotification(
        packageName: String, sourceName: String, key: String, title: String, body: String,
        postedAt: Long, zoneId: String, now: Long = System.currentTimeMillis(),
    ): Boolean {
        if (!acceptsPackage(packageName) || (title.isBlank() && body.isBlank())) return false
        if (title.length + body.length > MAX_TEXT) {
            recordExcluded()
            return false
        }
        val contentDigest = digest(title + "\u0000" + body)
        val db = store.writableDatabase
        var inserted = false
        db.beginTransaction()
        try {
            val deleted = db.rawQuery("SELECT 1 FROM life_tombstones WHERE notification_key=? AND digest=?",
                arrayOf(digest(key), contentDigest)).use { it.moveToFirst() }
            if (!deleted && acceptsPackage(packageName)) {
                val existing = db.rawQuery(
                    "SELECT id FROM life_records WHERE kind='notification' AND package_name=? AND notification_key=? AND active=1 LIMIT 1",
                    arrayOf(packageName, key),
                ).use { if (it.moveToFirst()) it.getString(0) else null }
                val id = existing ?: UUID.randomUUID().toString().also { newId ->
                    db.insertOrThrow("life_records", null, ContentValues().apply {
                        put("id", newId); put("kind", "notification"); put("package_name", packageName)
                        put("source_name", sourceName); put("notification_key", key); put("active", 1)
                        put("created_at", now); put("updated_at", now); put("processing", "pending")
                    })
                }
                val lastDigest = db.rawQuery("SELECT digest FROM life_versions WHERE record_id=? ORDER BY id DESC LIMIT 1",
                    arrayOf(id)).use { if (it.moveToFirst()) it.getString(0) else null }
                val versionId = if (lastDigest == contentDigest) -1L else db.insertOrThrow("life_versions", null, ContentValues().apply {
                    put("record_id", id); put("title", title); put("body", body); put("digest", contentDigest)
                    put("captured_at", now); put("posted_at", postedAt); put("zone_id", zoneId)
                })
                if (versionId != -1L) {
                    db.update("life_records", ContentValues().apply { put("updated_at", now); put("processing", "pending") },
                        "id=?", arrayOf(id))
                    val base = runCatching { Instant.ofEpochMilli(postedAt).atZone(ZoneId.of(zoneId)).toLocalDate() }.getOrNull()
                    addCandidates(db, id, versionId, title + "\n" + body, base)
                    inserted = true
                }
            }
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        if (inserted) { LifeSignals.changed(); schedule(app) }
        return inserted
    }

    @Synchronized fun notificationRemoved(key: String, lockdown: Boolean) {
        val db = store.writableDatabase
        if (lockdown) db.delete("life_records", "kind='notification' AND notification_key=?", arrayOf(key))
        else {
            db.update("life_records", ContentValues().apply { put("active", 0) },
                "kind='notification' AND notification_key=?", arrayOf(key))
            db.delete("life_tombstones", "notification_key=?", arrayOf(digest(key)))
        }
        LifeSignals.changed()
    }

    @Synchronized fun reconcileActive(keys: Set<String>, allActiveKeys: Set<String>) {
        val db = store.writableDatabase
        val active = db.rawQuery("SELECT id, notification_key FROM life_records WHERE kind='notification' AND active=1", null)
            .use { c -> buildList { while (c.moveToNext()) add(c.getString(0) to c.getString(1)) } }
        active.filter { it.second !in keys }.forEach { (id, _) ->
            db.update("life_records", ContentValues().apply { put("active", 0) }, "id=?", arrayOf(id))
        }
        val activeHashes = allActiveKeys.map(::digest).toSet()
        val tombstones = db.rawQuery("SELECT DISTINCT notification_key FROM life_tombstones", null).use { c ->
            buildList { while (c.moveToNext()) add(c.getString(0)) }
        }
        tombstones.filter { it !in activeHashes }.forEach { db.delete("life_tombstones", "notification_key=?", arrayOf(it)) }
        LifeSignals.changed()
    }

    fun excludedCount(): Int = prefs.getInt("excluded", 0)

    @Synchronized fun recordExcluded() {
        prefs.edit().putInt("excluded", prefs.getInt("excluded", 0) + 1).apply()
        LifeSignals.changed()
    }

    private fun addCandidates(db: SQLiteDatabase, recordId: String, versionId: Long?, text: String, base: LocalDate?) {
        LifeRules.proposals(text, base).forEach { (kind, sentence, date) ->
            val fingerprint = digest(recordId + "|" + (versionId ?: 0) + "|" + kind + "|" + sentence)
            db.insertWithOnConflict("life_candidates", null, ContentValues().apply {
                put("record_id", recordId); if (versionId != null) put("version_id", versionId)
                put("kind", kind); put("title", sentence); put("due_date", date?.toString())
                put("evidence", sentence); put("fingerprint", fingerprint)
            }, SQLiteDatabase.CONFLICT_IGNORE)
        }
    }

    @Synchronized fun cleanupExpired(now: Long = System.currentTimeMillis()) {
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            db.rawQuery("""SELECT v.id, v.digest, r.notification_key, r.active FROM life_versions v
                JOIN life_records r ON r.id=v.record_id WHERE v.captured_at<=?""",
                arrayOf((now - RETENTION_MS).toString())).use { c ->
                while (c.moveToNext()) {
                    if (c.getInt(3) != 0) db.insertWithOnConflict("life_tombstones", null, ContentValues().apply {
                        put("notification_key", digest(c.getString(2))); put("digest", c.getString(1))
                        put("expires_at", Long.MAX_VALUE)
                    }, SQLiteDatabase.CONFLICT_REPLACE)
                }
            }
            db.delete("life_versions", "captured_at<=?", arrayOf((now - RETENTION_MS).toString()))
            db.execSQL("DELETE FROM life_records WHERE kind='notification' AND NOT EXISTS (SELECT 1 FROM life_versions WHERE record_id=life_records.id)")
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
    }

    fun nextExpiry(): Long? = store.readableDatabase.rawQuery("SELECT MIN(captured_at) FROM life_versions", null).use {
        if (it.moveToFirst() && !it.isNull(0)) it.getLong(0) + RETENTION_MS else null
    }

    @Synchronized fun deleteRecord(id: String) {
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            val key = db.rawQuery("SELECT notification_key,active FROM life_records WHERE id=?", arrayOf(id)).use {
                if (it.moveToFirst() && !it.isNull(0) && it.getInt(1) != 0) it.getString(0) else null
            }
            if (key != null) db.rawQuery("SELECT digest FROM life_versions WHERE record_id=?", arrayOf(id)).use { c ->
                while (c.moveToNext()) db.insertWithOnConflict("life_tombstones", null, ContentValues().apply {
                    put("notification_key", digest(key)); put("digest", c.getString(0))
                    put("expires_at", Long.MAX_VALUE)
                }, SQLiteDatabase.CONFLICT_REPLACE)
            }
            db.delete("life_records", "id=?", arrayOf(id))
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
    }

    fun deleteNotifications() {
        val ids = store.readableDatabase.rawQuery("SELECT id FROM life_records WHERE kind='notification'", null).use { c ->
            buildList { while (c.moveToNext()) add(c.getString(0)) }
        }
        ids.forEach(::deleteRecord)
    }

    fun versions(recordId: String, now: Long = System.currentTimeMillis()): List<LifeVersion> =
        store.readableDatabase.rawQuery(
            "SELECT id,record_id,title,body,captured_at,posted_at,zone_id,processing FROM life_versions WHERE record_id=? AND captured_at>? ORDER BY id",
            arrayOf(recordId, (now - RETENTION_MS).toString()),
        ).use { c -> buildList {
            while (c.moveToNext()) add(LifeVersion(c.getLong(0), c.getString(1), c.getString(2), c.getString(3),
                c.getLong(4), c.getLong(5), c.getString(6), c.getString(7)))
        } }

    fun records(now: Long = System.currentTimeMillis()): List<LifeRecord> {
        cleanupExpired(now)
        refreshLinkedDocuments()
        val db = store.readableDatabase
        return db.rawQuery("""SELECT id,kind,title,body,created_at,updated_at,source_name,note_id,package_name,
            cluster_vector,search_vectors,processing,error,truth_order FROM life_records ORDER BY updated_at DESC""", null).use { c ->
            buildList {
                while (c.moveToNext()) {
                    val id = c.getString(0)
                    val kind = c.getString(1)
                    var title = c.getString(2)
                    var body = c.getString(3)
                    val created = c.getLong(4)
                    var updated = c.getLong(5)
                    var source = if (c.isNull(6)) null else c.getString(6)
                    val noteId = if (c.isNull(7)) null else c.getString(7)
                    val packageName = if (c.isNull(8)) null else c.getString(8)
                    var vector = if (c.isNull(9)) null else FloatCodec.decodeOne(c.getBlob(9))
                    var search = if (c.isNull(10)) emptyList() else FloatCodec.decode(c.getBlob(10))
                    var processing = c.getString(11)
                    var error = if (c.isNull(12)) null else c.getString(12)
                    if (kind == "document") {
                        val note = noteId?.let(store::note) ?: continue
                        title = note.title; body = note.text; updated = note.updatedAt
                        source = note.source?.fileName
                        vector = note.clusterVector; search = note.searchVectors; processing = "ready"; error = null
                    } else if (kind == "notification") {
                        val allVectors = mutableListOf<FloatArray>()
                        val allSearch = mutableListOf<FloatArray>()
                        val texts = mutableListOf<String>()
                        var lastTitle = ""
                        var pending = false
                        var failed = false
                        db.rawQuery("""SELECT title,body,cluster_vector,search_vectors,processing,error
                            FROM life_versions WHERE record_id=? AND captured_at>? ORDER BY id""",
                            arrayOf(id, (now - RETENTION_MS).toString())).use { v ->
                            while (v.moveToNext()) {
                                lastTitle = v.getString(0)
                                texts += listOf(v.getString(0), v.getString(1)).filter(String::isNotBlank).joinToString("\n")
                                when (v.getString(4)) {
                                    "ready" -> {
                                        if (!v.isNull(2)) allVectors += FloatCodec.decodeOne(v.getBlob(2))
                                        if (!v.isNull(3)) allSearch += FloatCodec.decode(v.getBlob(3))
                                    }
                                    "failed" -> { failed = true; error = if (v.isNull(5)) "처리 실패" else v.getString(5) }
                                    else -> pending = true
                                }
                            }
                        }
                        if (texts.isEmpty()) continue
                        title = lastTitle.ifBlank { source ?: "알림" }
                        body = texts.joinToString("\n\n")
                        processing = if (failed) "failed" else if (pending || allVectors.isEmpty()) "pending" else "ready"
                        vector = if (processing == "ready") normalizedMean(allVectors) else null
                        search = if (processing == "ready") allSearch else emptyList()
                    }
                    val sourceForOrder = if (kind == "notification") source else null
                    add(LifeRecord(id, kind, title, body, created, updated, source, noteId, packageName,
                        LifeRules.orderKey(title, body, sourceForOrder), search, vector, processing, error,
                        if (c.isNull(13)) null else c.getString(13)))
                }
            }
        }
    }

    private fun normalizedMean(vectors: List<FloatArray>): FloatArray {
        val result = FloatArray(vectors.first().size) { i -> vectors.sumOf { it[i].toDouble() }.div(vectors.size).toFloat() }
        val norm = sqrt(result.sumOf { it.toDouble() * it })
        return if (norm == 0.0) result else FloatArray(result.size) { (result[it] / norm).toFloat() }
    }

    private fun refreshLinkedDocuments() {
        val db = store.writableDatabase
        val links = db.rawQuery("SELECT id,note_id,note_updated_at FROM life_records WHERE kind='document'", null).use { c ->
            buildList { while (c.moveToNext()) add(Triple(c.getString(0), c.getString(1), if (c.isNull(2)) 0L else c.getLong(2))) }
        }
        links.forEach { (id, noteId, previous) ->
            val note = store.note(noteId) ?: return@forEach
            if (note.updatedAt != previous) {
                db.beginTransaction()
                try {
                    db.delete("life_candidates", "record_id=? AND state='pending'", arrayOf(id))
                    addCandidates(db, id, null, note.title + "\n" + note.text, null)
                    db.update("life_records", ContentValues().apply {
                        put("note_updated_at", note.updatedAt); put("updated_at", note.updatedAt)
                    }, "id=?", arrayOf(id))
                    db.setTransactionSuccessful()
                } finally { db.endTransaction() }
            }
        }
    }

    data class Pending(val recordId: String, val versionId: Long?, val title: String, val body: String, val revision: Int)

    fun nextPending(now: Long = System.currentTimeMillis()): Pending? {
        val db = store.readableDatabase
        db.rawQuery("""SELECT record_id,id,title,body FROM life_versions
            WHERE processing='pending' AND captured_at>? ORDER BY id LIMIT 1""",
            arrayOf((now - RETENTION_MS).toString())).use { c ->
            if (c.moveToFirst()) return Pending(c.getString(0), c.getLong(1), c.getString(2), c.getString(3), 0)
        }
        return db.rawQuery("""SELECT id,title,body,revision FROM life_records
            WHERE kind IN ('memo','share') AND processing='pending' ORDER BY created_at LIMIT 1""", null).use { c ->
            if (c.moveToFirst()) Pending(c.getString(0), null, c.getString(1), c.getString(2), c.getInt(3)) else null
        }
    }

    fun completePending(item: Pending, vectors: NoteVectors) {
        val values = ContentValues().apply {
            put("cluster_vector", FloatCodec.encode(listOf(vectors.cluster)))
            put("search_vectors", FloatCodec.encode(vectors.searchChunks))
            put("processing", "ready"); putNull("error")
        }
        if (item.versionId != null) {
            store.writableDatabase.update("life_versions", values, "id=? AND processing='pending' AND captured_at>?",
                arrayOf(item.versionId.toString(), (System.currentTimeMillis() - RETENTION_MS).toString()))
        } else {
            store.writableDatabase.update("life_records", values, "id=? AND revision=? AND processing='pending'",
                arrayOf(item.recordId, item.revision.toString()))
        }
        LifeSignals.changed()
    }

    fun failPending(item: Pending, reason: String): Boolean {
        val table = if (item.versionId == null) "life_records" else "life_versions"
        val key = item.versionId?.toString() ?: item.recordId
        val where = if (item.versionId == null) "id=? AND revision=? AND processing='pending'" else "id=? AND processing='pending'"
        val args = if (item.versionId == null) arrayOf(key, item.revision.toString()) else arrayOf(key)
        val db = store.writableDatabase
        val attempts = db.rawQuery("SELECT attempts FROM $table WHERE $where", args).use {
            if (it.moveToFirst()) it.getInt(0) + 1 else return false
        }
        val retry = attempts < 2
        db.update(table, ContentValues().apply {
            put("attempts", attempts); put("processing", if (retry) "pending" else "failed")
            put("error", reason.take(200))
        }, where, args)
        LifeSignals.changed()
        return retry
    }

    fun retryFailures() {
        val db = store.writableDatabase
        val values = ContentValues().apply { put("processing", "pending"); put("attempts", 0); putNull("error") }
        db.update("life_versions", values, "processing='failed'", null)
        db.update("life_records", values, "processing='failed' AND kind IN ('memo','share')", null)
        LifeSignals.changed()
        schedule(app)
    }

    fun updateText(id: String, title: String, body: String) {
        val cleanTitle = title.trim()
        val cleanBody = body.trim()
        require(cleanTitle.isNotEmpty() && cleanBody.isNotEmpty()) { "제목과 내용을 입력해 주세요." }
        require(cleanBody.length <= MAX_TEXT) { "메모와 공유 글은 20,000자까지 저장할 수 있습니다." }
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            val now = System.currentTimeMillis()
            db.execSQL("""UPDATE life_records SET title=?,body=?,updated_at=?,source_time=CASE WHEN kind='memo' THEN ? ELSE source_time END,
                revision=revision+1,cluster_vector=NULL,search_vectors=NULL,processing='pending',error=NULL,attempts=0
                WHERE id=? AND kind IN ('memo','share')""", arrayOf<Any>(cleanTitle, cleanBody, now, now, id))
            db.delete("life_candidates", "record_id=? AND state='pending'", arrayOf(id))
            val base = db.rawQuery("SELECT source_time,zone_id FROM life_records WHERE id=?", arrayOf(id)).use { c ->
                if (c.moveToFirst() && !c.isNull(0)) runCatching {
                    Instant.ofEpochMilli(c.getLong(0)).atZone(ZoneId.of(c.getString(1))).toLocalDate()
                }.getOrNull() else null
            }
            addCandidates(db, id, null, cleanTitle + "\n" + cleanBody, base)
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
        schedule(app)
    }

    fun candidates(): List<LifeCandidate> {
        cleanupExpired()
        return store.readableDatabase.rawQuery("""SELECT c.id,c.record_id,c.version_id,c.kind,c.title,c.due_date,c.evidence
            FROM life_candidates c LEFT JOIN life_versions v ON v.id=c.version_id
            WHERE c.state='pending' AND (c.version_id IS NULL OR v.captured_at>?) ORDER BY c.id DESC""",
            arrayOf((System.currentTimeMillis() - RETENTION_MS).toString())).use { c -> buildList {
            while (c.moveToNext()) add(LifeCandidate(c.getLong(0), c.getString(1), if (c.isNull(2)) null else c.getLong(2),
                c.getString(3), c.getString(4), if (c.isNull(5)) null else c.getString(5), c.getString(6)))
        } }
    }

    fun approveCandidate(candidate: LifeCandidate, kind: String, title: String, date: String?) {
        require(kind in setOf("task", "event")) { "일정 또는 할 일을 선택해 주세요." }
        val clean = title.trim()
        require(clean.isNotEmpty()) { "제목을 입력해 주세요." }
        val due = date?.trim()?.takeIf(String::isNotEmpty)
        if (due != null) require(runCatching { LocalDate.parse(due) }.isSuccess) { "날짜는 YYYY-MM-DD 형식으로 입력해 주세요." }
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            val sourceKind = db.rawQuery("SELECT kind FROM life_records WHERE id=?", arrayOf(candidate.recordId)).use {
                if (it.moveToFirst()) it.getString(0) else null
            }
            check(sourceKind != null) { "원문이 삭제됐습니다." }
            if (candidate.versionId != null) {
                check(db.rawQuery("SELECT 1 FROM life_versions WHERE id=? AND captured_at>?",
                    arrayOf(candidate.versionId.toString(), (System.currentTimeMillis() - RETENTION_MS).toString())).use { it.moveToFirst() }) {
                    "알림 원문의 보관 기간이 끝났습니다."
                }
            }
            val changed = db.update("life_candidates", ContentValues().apply { put("state", "accepted") },
                "id=? AND state='pending'", arrayOf(candidate.id.toString()))
            check(changed == 1) { "이미 처리된 후보입니다." }
            db.insertOrThrow("life_tasks", null, ContentValues().apply {
                put("id", UUID.randomUUID().toString()); put("source_id", candidate.recordId)
                if (candidate.versionId != null) put("source_version_id", candidate.versionId)
                put("source_kind", sourceKind); put("kind", kind); put("title", clean)
                put("due_date", due); put("created_at", System.currentTimeMillis())
            })
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
    }

    fun dismissCandidate(id: Long) {
        store.writableDatabase.update("life_candidates", ContentValues().apply { put("state", "dismissed") },
            "id=?", arrayOf(id.toString()))
        LifeSignals.changed()
    }

    fun tasks(): List<LifeTask> {
        cleanupExpired()
        return store.readableDatabase.rawQuery("""SELECT id,source_id,kind,title,due_date,done,source_kind,source_version_id
            FROM life_tasks ORDER BY done,CASE WHEN due_date IS NULL THEN 1 ELSE 0 END,due_date,created_at DESC""", null).use { c ->
            buildList {
                while (c.moveToNext()) {
                    val source = if (c.isNull(1)) null else c.getString(1)
                    val sourceKind = if (c.isNull(6)) null else c.getString(6)
                    val expired = sourceKind != null && (source == null || (sourceKind == "notification" && c.isNull(7)))
                    add(LifeTask(c.getString(0), source, c.getString(2), c.getString(3),
                        if (c.isNull(4)) null else c.getString(4), c.getInt(5) != 0, expired))
                }
            }
        }
    }

    fun saveTask(id: String?, kind: String, title: String, date: String?) {
        require(kind == "task" || kind == "event") { "할 일 또는 일정을 선택해 주세요." }
        val clean = title.trim()
        require(clean.isNotEmpty()) { "제목을 입력해 주세요." }
        val due = date?.trim()?.takeIf(String::isNotEmpty)
        if (due != null) require(runCatching { LocalDate.parse(due) }.isSuccess) { "날짜는 YYYY-MM-DD 형식으로 입력해 주세요." }
        val values = ContentValues().apply { put("kind", kind); put("title", clean); put("due_date", due) }
        if (id == null) {
            values.put("id", UUID.randomUUID().toString()); values.put("created_at", System.currentTimeMillis())
            store.writableDatabase.insertOrThrow("life_tasks", null, values)
        } else store.writableDatabase.update("life_tasks", values, "id=?", arrayOf(id))
        LifeSignals.changed()
    }

    fun setTaskDone(id: String, done: Boolean) {
        store.writableDatabase.update("life_tasks", ContentValues().apply { put("done", if (done) 1 else 0) }, "id=?", arrayOf(id))
        LifeSignals.changed()
    }

    fun deleteTask(id: String) {
        store.writableDatabase.delete("life_tasks", "id=?", arrayOf(id))
        LifeSignals.changed()
    }

    fun setTruthOrder(id: String, label: String) {
        store.writableDatabase.update("life_records", ContentValues().apply {
            put("truth_order", label.trim().takeIf(String::isNotEmpty))
        }, "id=?", arrayOf(id))
        LifeSignals.changed()
    }

    fun manualGroups(kind: String): List<LifeManualGroup> {
        val db = store.readableDatabase
        return db.rawQuery("SELECT id,kind,title FROM life_groups WHERE kind=? ORDER BY created_at", arrayOf(kind)).use { c ->
            buildList {
                while (c.moveToNext()) {
                    val id = c.getString(0)
                    val members = db.rawQuery("SELECT record_id,membership FROM life_memberships WHERE group_id=?", arrayOf(id)).use { m ->
                        buildList { while (m.moveToNext()) add(m.getString(0) to m.getString(1)) }
                    }
                    add(LifeManualGroup(id, c.getString(1), c.getString(2), members))
                }
            }
        }
    }

    fun createGroup(kind: String, title: String, recordIds: List<String>): String {
        require(kind in setOf("topic", "order"))
        val id = UUID.randomUUID().toString()
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            db.insertOrThrow("life_groups", null, ContentValues().apply {
                put("id", id); put("kind", kind); put("title", title.trim().ifEmpty { "이름 미정" })
                put("created_at", System.currentTimeMillis())
            })
            recordIds.distinct().forEach { recordId -> assignMember(db, id, kind, recordId, "primary") }
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
        return id
    }

    fun renameGroup(id: String, title: String) {
        require(title.isNotBlank()) { "이름을 입력해 주세요." }
        store.writableDatabase.update("life_groups", ContentValues().apply { put("title", title.trim()) }, "id=?", arrayOf(id))
        LifeSignals.changed()
    }

    fun addMember(groupId: String, recordId: String, extra: Boolean = false) {
        val db = store.writableDatabase
        val kind = db.rawQuery("SELECT kind FROM life_groups WHERE id=?", arrayOf(groupId)).use {
            if (it.moveToFirst()) it.getString(0) else error("묶음을 찾을 수 없습니다.")
        }
        db.beginTransaction()
        try {
            assignMember(db, groupId, kind, recordId, if (extra && kind == "topic") "extra" else "primary")
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        LifeSignals.changed()
    }

    private fun assignMember(db: SQLiteDatabase, groupId: String, kind: String, recordId: String, membership: String) {
        if (membership == "primary") db.execSQL("""DELETE FROM life_memberships WHERE record_id=? AND membership='primary'
            AND group_id IN (SELECT id FROM life_groups WHERE kind=?)""", arrayOf(recordId, kind))
        db.insertWithOnConflict("life_memberships", null, ContentValues().apply {
            put("group_id", groupId); put("record_id", recordId); put("membership", membership)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun removeMember(groupId: String, recordId: String) {
        val db = store.writableDatabase
        val primaryTopic = db.rawQuery("""SELECT 1 FROM life_memberships m JOIN life_groups g ON g.id=m.group_id
            WHERE m.group_id=? AND m.record_id=? AND m.membership='primary' AND g.kind='topic'""",
            arrayOf(groupId, recordId)).use { it.moveToFirst() }
        if (primaryTopic) {
            createGroup("topic", "분리한 주제", listOf(recordId))
            return
        }
        db.delete("life_memberships", "group_id=? AND record_id=?", arrayOf(groupId, recordId))
        LifeSignals.changed()
    }

    fun saveRun(threshold: Double, dataHash: String, modelHash: String, model: List<List<String>>, hybrid: List<List<String>>) {
        fun encode(groups: List<List<String>>) = JSONArray().apply { groups.forEach { put(JSONArray(it)) } }.toString()
        store.writableDatabase.insertOrThrow("life_runs", null, ContentValues().apply {
            put("created_at", System.currentTimeMillis()); put("threshold", threshold); put("data_hash", dataHash)
            put("model_hash", modelHash); put("model_groups", encode(model)); put("hybrid_groups", encode(hybrid))
        })
    }

    fun setReviewed(dataset: String, itemId: String, reviewed: Boolean) {
        check(!(dataset == LifeSynthetic.DEV && trialActive())) { "7일 시험 중에는 개발 자료 검토를 변경할 수 없습니다." }
        store.writableDatabase.insertWithOnConflict("life_reviews", null, ContentValues().apply {
            put("dataset", dataset); put("item_id", itemId); put("reviewed", if (reviewed) 1 else 0)
        }, SQLiteDatabase.CONFLICT_REPLACE)
        LifeSignals.changed()
    }

    fun reviewedIds(dataset: String): Set<String> = store.readableDatabase.rawQuery(
        "SELECT item_id FROM life_reviews WHERE dataset=? AND reviewed=1", arrayOf(dataset),
    ).use { c -> buildSet { while (c.moveToNext()) add(c.getString(0)) } }

    fun saveEvaluation(dataset: String, thresholdValue: Double, modelHash: String, model: LifePairMetrics, hybrid: LifePairMetrics) {
        check(dataset == LifeSynthetic.TEST && reviewedIds(dataset).containsAll(LifeSynthetic.orders(dataset).keys)) { "평가 자료 검토를 먼저 마쳐 주세요." }
        check(threshold() != null && modelHash == currentModelHash()) { "현재 모델의 기준값을 먼저 정해 주세요." }
        check(model.positivePairs >= 50 && model.negativePairs >= 50 && hybrid.positivePairs == model.positivePairs && hybrid.negativePairs == model.negativePairs) {
            "검토한 관련·무관 쌍이 각각 50개 이상 필요합니다."
        }
        store.writableDatabase.insertOrThrow("life_evaluations", null, ContentValues().apply {
            put("created_at", System.currentTimeMillis()); put("dataset", dataset); put("model_hash", modelHash)
            put("threshold", thresholdValue); put("positives", model.positivePairs); put("negatives", model.negativePairs)
            put("model_precision", requireNotNull(model.precision)); put("model_recall", requireNotNull(model.recall))
            put("model_f1", requireNotNull(model.f1)); put("model_false_merges", model.falsePositive)
            put("hybrid_precision", requireNotNull(hybrid.precision)); put("hybrid_recall", requireNotNull(hybrid.recall))
            put("hybrid_f1", requireNotNull(hybrid.f1)); put("hybrid_false_merges", hybrid.falsePositive)
        })
        LifeSignals.changed()
    }

    fun lastEvaluation(): LifeEvaluationResult? = store.readableDatabase.rawQuery("""SELECT dataset,threshold,positives,negatives,
        model_precision,model_recall,model_f1,model_false_merges,hybrid_precision,hybrid_recall,hybrid_f1,hybrid_false_merges,model_hash,created_at
        FROM life_evaluations ORDER BY id DESC LIMIT 1""", null).use { c ->
        if (!c.moveToFirst()) null else LifeEvaluationResult(c.getString(0), c.getDouble(1), c.getLong(2), c.getLong(3),
            c.getDouble(4), c.getDouble(5), c.getDouble(6), c.getLong(7), c.getDouble(8), c.getDouble(9), c.getDouble(10),
            c.getLong(11), c.getString(12), c.getLong(13))
    }

    fun saveUsage(scenario: String, method: String, success: Boolean, seconds: Int?, issue: String) {
        store.writableDatabase.insertOrThrow("life_usage", null, ContentValues().apply {
            put("created_at", System.currentTimeMillis()); put("scenario", scenario); put("method", method)
            put("success", if (success) 1 else 0); if (seconds != null) put("seconds", seconds)
            put("issue", issue.trim())
        })
        LifeSignals.changed()
    }

    fun usageSummary(): List<String> = store.readableDatabase.rawQuery(
        "SELECT scenario,method,success,seconds,issue FROM life_usage ORDER BY created_at DESC", null,
    ).use { c -> buildList {
        while (c.moveToNext()) add(c.getString(0) + " · " + c.getString(1) + " · " +
            (if (c.getInt(2) != 0) "성공" else "실패") +
            (if (c.isNull(3)) "" else " · " + c.getInt(3) + "초") +
            (if (c.getString(4).isBlank()) "" else " · " + c.getString(4)))
    } }

    fun trialStart(): Long? = prefs.getLong("trial_start", 0L).takeIf { it > 0 }
    fun trialActive(now: Long = System.currentTimeMillis()): Boolean = trialStart()?.let { now < it + 7L * 24 * 60 * 60 * 1000 } ?: false
    fun startTrial() {
        check(threshold() != null) { "7일 시험 전에 개발 자료를 검토하고 기준값을 정해 주세요." }
        check(trialStart() == null) { "이미 7일 시험을 시작했습니다." }
        prefs.edit().putLong("trial_start", System.currentTimeMillis()).commit()
        LifeSignals.changed()
    }
}
