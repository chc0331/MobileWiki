package com.heecchoi.slmwikipoc

import android.content.ContentValues
import android.content.Context
import org.json.JSONArray
import java.io.File
import java.security.MessageDigest
import java.util.UUID
import kotlin.math.roundToInt

data class ExperimentItem(
    val id: String, val kind: String, val title: String, val body: String, val label: String?,
    val createdAt: Long, val updatedAt: Long, val revision: Long, val source: DocumentSource?,
    val vector: FloatArray?, val modelHash: String?, val pipelineVersion: String?, val scanError: String? = null,
) {
    fun note(vectorOverride: FloatArray = vector ?: FloatArray(768)): Note =
        Note(id, "simple-experiment", null, title, body, createdAt, updatedAt,
            listOf(vectorOverride), vectorOverride, label, source)
}

data class ExperimentQuality(
    val labeled: Int, val pairs: Long, val falseMerges: Long,
    val precision: Double?, val recall: Double?, val f1: Double?,
)

data class ExperimentRunPoint(val inputId: String, val groupId: Int, val groupSize: Int, val x: Float, val y: Float)
data class ScannedInputRecord(val id: String, val path: String, val size: Long, val modifiedAt: Long, val error: String?)

data class ExperimentRun(
    val signature: String, val modelHash: String, val threshold: Double, val createdAt: Long,
    val embeddingMs: Long, val clusteringMs: Long, val computed: Int, val reused: Int,
    val quality: ExperimentQuality, val points: List<ExperimentRunPoint>,
)

object ExperimentQualityCalculator {
    fun calculate(items: List<ExperimentItem>, groups: List<NoteGroup>, checkCancelled: () -> Unit = {}): ExperimentQuality {
        val byId = mutableMapOf<String, Int>()
        groups.forEach { group -> group.notes.forEach { note ->
            check(byId.put(note.id, group.id) == null) { "군집에 중복 항목이 있습니다." }
        } }
        require(byId.keys == items.map(ExperimentItem::id).toSet() && items.map(ExperimentItem::id).distinct().size == items.size) {
            "일부 항목의 군집 결과가 없습니다."
        }
        val labeled = items.filter { !it.label.isNullOrBlank() }
        val pairs = labeled.size.toLong() * (labeled.size - 1) / 2
        if (pairs == 0L) return ExperimentQuality(labeled.size, 0, 0, null, null, null)
        var tp = 0L
        var fp = 0L
        var fn = 0L
        for (i in 0 until labeled.size) {
            checkCancelled()
            for (j in i + 1 until labeled.size) {
                val sameTruth = labeled[i].label!!.trim() == labeled[j].label!!.trim()
                val sameGroup = byId.getValue(labeled[i].id) == byId.getValue(labeled[j].id)
                if (sameTruth && sameGroup) tp++ else if (sameTruth) fn++ else if (sameGroup) fp++
            }
        }
        val precision = if (tp + fp == 0L) 0.0 else tp.toDouble() / (tp + fp)
        val recall = if (tp + fn == 0L) 0.0 else tp.toDouble() / (tp + fn)
        val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
        return ExperimentQuality(labeled.size, pairs, fp, precision, recall, f1)
    }
}

class ExperimentStore(context: Context, private val store: LocalStore, preferences: String = "simple-experiment") {
    private val app = context.applicationContext
    private val prefs = app.getSharedPreferences(preferences, Context.MODE_PRIVATE)

    companion object {
        const val PIPELINE_VERSION = "clustering-full-body-v2"
        const val MAX_TEXT = 20_000
        fun signature(items: List<ExperimentItem>, threshold: Double, modelHash: String): String {
            val value = buildString {
                append(modelHash).append('|').append(PIPELINE_VERSION).append('|').append((threshold * 100).roundToInt())
                items.sortedBy(ExperimentItem::id).forEach { append('|').append(it.id).append(':').append(it.revision) }
            }
            return MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
                .joinToString("") { "%02x".format(it) }
        }
        fun autoTitle(body: String): String = body.lineSequence().firstOrNull { it.isNotBlank() }
            ?.trim()?.take(80).orEmpty().ifBlank { "제목 없음" }
    }

    @Synchronized fun threshold(): Double = prefs.getInt("threshold_steps", 5) / 100.0
    @Synchronized fun setThreshold(value: Double) {
        prefs.edit().putInt("threshold_steps", (value.coerceIn(0.0, 2.0) * 100).roundToInt()).apply()
    }

    @Synchronized fun items(): List<ExperimentItem> = store.readableDatabase.rawQuery(
        """SELECT id,kind,title,body,ground_truth,created_at,updated_at,revision,source_name,source_mime,
           source_path,page_texts,device_path,device_size,device_modified_at,cluster_vector,model_hash,pipeline_version,scan_error
           FROM experiment_inputs ORDER BY updated_at DESC,id""", null,
    ).use { c -> buildList {
        while (c.moveToNext()) {
            val sourceName = if (c.isNull(8)) null else c.getString(8)
            val pages = if (c.isNull(11)) emptyList() else JSONArray(c.getString(11)).let { json ->
                (0 until json.length()).map(json::getString)
            }
            val source = sourceName?.let {
                DocumentSource(it, if (c.isNull(9)) "application/octet-stream" else c.getString(9),
                    if (c.isNull(10)) null else c.getString(10), pages,
                    if (c.isNull(12)) null else c.getString(12),
                    if (c.isNull(13)) null else c.getLong(13), if (c.isNull(14)) null else c.getLong(14))
            }
            add(ExperimentItem(c.getString(0), c.getString(1), c.getString(2), c.getString(3),
                if (c.isNull(4)) null else c.getString(4), c.getLong(5), c.getLong(6), c.getLong(7), source,
                if (c.isNull(15)) null else FloatCodec.decodeOne(c.getBlob(15)),
                if (c.isNull(16)) null else c.getString(16), if (c.isNull(17)) null else c.getString(17),
                if (c.isNull(18)) null else c.getString(18)))
        }
    } }

    @Synchronized fun saveText(id: String?, kind: String, title: String, body: String) {
        require(kind == "text" || kind == "voice") { "입력 방식을 확인해 주세요." }
        val clean = body.trim()
        require(clean.isNotBlank()) { "내용을 입력해 주세요." }
        require(clean.length <= MAX_TEXT) { "직접 입력과 음성 입력은 20,000자까지 저장할 수 있습니다." }
        val cleanTitle = title.trim().ifBlank { autoTitle(clean) }
        val now = System.currentTimeMillis()
        val db = store.writableDatabase
        if (id == null) {
            db.insertOrThrow("experiment_inputs", null, ContentValues().apply {
                put("id", UUID.randomUUID().toString()); put("kind", kind); put("title", cleanTitle)
                put("body", clean); put("created_at", now); put("updated_at", now)
            })
        } else {
            db.beginTransaction()
            try {
                val count = db.update("experiment_inputs", ContentValues().apply {
                    put("title", cleanTitle); put("body", clean); put("updated_at", now)
                    putNull("cluster_vector"); putNull("model_hash"); putNull("pipeline_version")
                }, "id=? AND kind IN ('text','voice')", arrayOf(id))
                check(count == 1) { "수정할 입력을 찾지 못했습니다." }
                db.execSQL("UPDATE experiment_inputs SET revision=revision+1 WHERE id=?", arrayOf(id))
                db.setTransactionSuccessful()
            } finally { db.endTransaction() }
        }
    }

    @Synchronized fun addDocument(document: ParsedDocument, source: DocumentSource, label: String?) {
        val now = System.currentTimeMillis()
        store.writableDatabase.insertOrThrow("experiment_inputs", null, ContentValues().apply {
            put("id", UUID.randomUUID().toString()); put("kind", "document")
            put("title", document.title); put("body", document.text); put("ground_truth", label?.trim()?.takeIf(String::isNotBlank))
            put("created_at", now); put("updated_at", now); putSource(source)
        })
    }

    // Returns true for a new path and false for an existing path that was updated.
    @Synchronized fun saveScanned(document: ParsedDocument, source: DocumentSource, legacyLabel: String?): Boolean {
        val path = requireNotNull(source.devicePath)
        val db = store.writableDatabase
        val oldId = db.rawQuery("SELECT id FROM experiment_inputs WHERE device_path=?", arrayOf(path)).use {
            if (it.moveToFirst()) it.getString(0) else null
        }
        val now = System.currentTimeMillis()
        val values = ContentValues().apply {
            put("title", document.title); put("body", document.text); put("updated_at", now)
            putSource(source); putNull("cluster_vector"); putNull("model_hash"); putNull("pipeline_version"); putNull("scan_error")
        }
        if (oldId == null) {
            values.put("id", UUID.randomUUID().toString()); values.put("kind", "document")
            values.put("ground_truth", legacyLabel?.trim()?.takeIf(String::isNotBlank)); values.put("created_at", now)
            db.insertOrThrow("experiment_inputs", null, values)
            return true
        }
        db.beginTransaction()
        try {
            db.update("experiment_inputs", values, "id=?", arrayOf(oldId))
            db.execSQL("UPDATE experiment_inputs SET revision=revision+1 WHERE id=?", arrayOf(oldId))
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        return false
    }

    private fun ContentValues.putSource(source: DocumentSource) {
        put("source_name", source.fileName); put("source_mime", source.mimeType); put("source_path", source.storedPath)
        put("page_texts", JSONArray(source.pageTexts).toString()); put("device_path", source.devicePath)
        put("device_size", source.deviceSize); put("device_modified_at", source.deviceModifiedAt)
    }

    @Synchronized fun scanned(): List<ScannedInputRecord> = store.readableDatabase.rawQuery(
        "SELECT id,device_path,device_size,device_modified_at,scan_error FROM experiment_inputs WHERE device_path IS NOT NULL", null,
    ).use { c -> buildList {
        while (c.moveToNext()) add(ScannedInputRecord(c.getString(0), c.getString(1), c.getLong(2), c.getLong(3), if (c.isNull(4)) null else c.getString(4)))
    } }

    @Synchronized fun markScanError(id: String, reason: String) {
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            db.update("experiment_inputs", ContentValues().apply {
                put("scan_error", reason); put("updated_at", System.currentTimeMillis())
                putNull("cluster_vector"); putNull("model_hash"); putNull("pipeline_version")
            }, "id=?", arrayOf(id))
            db.execSQL("UPDATE experiment_inputs SET revision=revision+1 WHERE id=?", arrayOf(id))
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
    }

    @Synchronized fun removeMissingScanned(ids: List<String>, checkCancelled: () -> Unit = {}): Int {
        val db = store.writableDatabase
        var removed = 0
        db.beginTransaction()
        try {
            ids.forEach { id ->
                checkCancelled()
                removed += db.delete("experiment_inputs", "id=? AND device_path IS NOT NULL", arrayOf(id))
            }
            checkCancelled()
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        return removed
    }

    @Synchronized fun setLabel(id: String, label: String) {
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            check(db.update("experiment_inputs", ContentValues().apply {
                put("ground_truth", label.trim().takeIf(String::isNotEmpty)); put("updated_at", System.currentTimeMillis())
            }, "id=?", arrayOf(id)) == 1) { "항목을 찾지 못했습니다." }
            db.execSQL("UPDATE experiment_inputs SET revision=revision+1 WHERE id=?", arrayOf(id))
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
    }

    @Synchronized fun remove(id: String) {
        val path = store.readableDatabase.rawQuery("SELECT source_path FROM experiment_inputs WHERE id=?", arrayOf(id)).use {
            if (!it.moveToFirst() || it.isNull(0)) null else it.getString(0)
        }
        path?.let {
            val file = File(it)
            check(!file.exists() || file.delete()) { "저장된 PDF 사본을 삭제하지 못했습니다." }
        }
        store.writableDatabase.delete("experiment_inputs", "id=?", arrayOf(id))
    }

    @Synchronized fun clear() {
        val paths = store.readableDatabase.rawQuery("SELECT source_path FROM experiment_inputs WHERE source_path IS NOT NULL", null).use { c ->
            buildList { while (c.moveToNext()) add(c.getString(0)) }
        }
        val failures = paths.count { path -> File(path).let { it.exists() && !it.delete() } }
        check(failures == 0) { "PDF 사본 ${failures}개를 삭제하지 못했습니다." }
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            db.delete("experiment_inputs", null, null)
            db.delete("experiment_run", null, null)
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
    }

    @Synchronized fun cacheVector(item: ExperimentItem, vector: FloatArray, modelHash: String): Boolean {
        return store.writableDatabase.update("experiment_inputs", ContentValues().apply {
            put("cluster_vector", FloatCodec.encode(listOf(vector))); put("model_hash", modelHash); put("pipeline_version", PIPELINE_VERSION)
        }, "id=? AND revision=?", arrayOf(item.id, item.revision.toString())) == 1
    }

    @Synchronized fun saveRun(
        expectedSignature: String, modelHash: String, threshold: Double, groups: List<NoteGroup>, projected: List<ProjectedDocument>,
        quality: ExperimentQuality, embeddingMs: Long, clusteringMs: Long, computed: Int, reused: Int, checkCancelled: () -> Unit = {},
    ): Boolean {
        val current = items()
        if (signature(current, this.threshold(), modelHash) != expectedSignature || (this.threshold() * 100).roundToInt() != (threshold * 100).roundToInt()) return false
        val groupById = groups.flatMap { group -> group.notes.map { it.id to group.id } }.toMap()
        val groupSizes = groups.associate { it.id to it.notes.size }
        require(groupById.keys == current.map(ExperimentItem::id).toSet() && projected.size == groupById.size && projected.map(ProjectedDocument::noteId).toSet() == groupById.keys) {
            "군집 또는 지도에서 항목이 누락되었습니다."
        }
        val db = store.writableDatabase
        db.beginTransaction()
        try {
            checkCancelled()
            db.delete("experiment_run", null, null)
            db.insertOrThrow("experiment_run", null, ContentValues().apply {
                put("id", 1); put("signature", expectedSignature); put("model_hash", modelHash); put("threshold", threshold)
                put("created_at", System.currentTimeMillis()); put("embedding_ms", embeddingMs); put("clustering_ms", clusteringMs)
                put("computed", computed); put("reused", reused); put("labeled_count", quality.labeled); put("labeled_pairs", quality.pairs)
                put("false_merges", quality.falseMerges); put("precision", quality.precision); put("recall", quality.recall); put("f1", quality.f1)
            })
            projected.forEach { point ->
                checkCancelled()
                db.insertOrThrow("experiment_run_items", null, ContentValues().apply {
                    put("run_id", 1); put("input_id", point.noteId); put("group_id", groupById.getValue(point.noteId))
                    put("group_size", groupSizes.getValue(groupById.getValue(point.noteId)))
                    put("x", point.x); put("y", point.y)
                })
            }
            checkCancelled()
            db.setTransactionSuccessful()
        } finally { db.endTransaction() }
        return true
    }

    @Synchronized fun lastRun(): ExperimentRun? {
        val db = store.readableDatabase
        return db.rawQuery("""SELECT signature,model_hash,threshold,created_at,embedding_ms,clustering_ms,computed,reused,
            labeled_count,labeled_pairs,false_merges,precision,recall,f1 FROM experiment_run WHERE id=1""", null).use { c ->
            if (!c.moveToFirst()) return@use null
            val quality = ExperimentQuality(c.getInt(8), c.getLong(9), c.getLong(10),
                if (c.isNull(11)) null else c.getDouble(11), if (c.isNull(12)) null else c.getDouble(12),
                if (c.isNull(13)) null else c.getDouble(13))
            val points = db.rawQuery("SELECT input_id,group_id,group_size,x,y FROM experiment_run_items WHERE run_id=1 ORDER BY group_id,input_id", null).use { p ->
                buildList { while (p.moveToNext()) add(ExperimentRunPoint(p.getString(0), p.getInt(1), p.getInt(2), p.getFloat(3), p.getFloat(4))) }
            }
            ExperimentRun(c.getString(0), c.getString(1), c.getDouble(2), c.getLong(3), c.getLong(4), c.getLong(5), c.getInt(6), c.getInt(7), quality, points)
        }
    }
}
