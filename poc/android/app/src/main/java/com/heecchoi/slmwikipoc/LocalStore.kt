package com.heecchoi.slmwikipoc

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer
import java.nio.ByteOrder

data class Note(
    val id: String,
    val workspace: String,
    val eventId: String?,
    val title: String,
    val text: String,
    val createdAt: Long,
    val updatedAt: Long,
    val searchVectors: List<FloatArray>,
    val clusterVector: FloatArray,
)

data class NoteInput(val id: String, val eventId: String?, val title: String, val text: String)
data class DailyUsageSummary(val day: String, val positive: Int, val negative: Int, val coldP95Ms: Double?, val warmP95Ms: Double?, val failedRuns: Int)

object FloatCodec {
    fun encode(vectors: List<FloatArray>): ByteArray {
        val count = vectors.sumOf { it.size }
        val buffer = ByteBuffer.allocate(count * Float.SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        vectors.forEach { vector -> vector.forEach(buffer::putFloat) }
        return buffer.array()
    }

    fun decode(bytes: ByteArray, dimensions: Int = 768): List<FloatArray> {
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val values = FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float }
        require(values.isNotEmpty() && values.size % dimensions == 0) { "Invalid cached embedding" }
        return values.asList().chunked(dimensions).map { it.toFloatArray() }
    }

    fun decodeOne(bytes: ByteArray): FloatArray = decode(bytes).single()
}

class LocalStore(context: Context) : SQLiteOpenHelper(context, "notes-poc.db", null, 1) {
    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE notes (
                id TEXT PRIMARY KEY NOT NULL,
                workspace TEXT NOT NULL,
                event_id TEXT,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                search_vectors BLOB NOT NULL,
                cluster_vector BLOB NOT NULL
            )""".trimIndent(),
        )
        db.execSQL("CREATE INDEX notes_workspace_updated ON notes(workspace, updated_at DESC)")
        db.execSQL(
            """CREATE TABLE feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_type TEXT NOT NULL,
                item_id TEXT NOT NULL,
                decision INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(item_type, item_id)
            )""".trimIndent(),
        )
        db.execSQL(
            """CREATE TABLE eval_embeddings (
                doc_id TEXT PRIMARY KEY NOT NULL,
                model_hash TEXT NOT NULL,
                search_vectors BLOB NOT NULL,
                cluster_vector BLOB NOT NULL
            )""".trimIndent(),
        )
        db.execSQL(
            """CREATE TABLE performance_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                cold_p95_ms REAL NOT NULL,
                warm_p95_ms REAL NOT NULL,
                cold_failures INTEGER NOT NULL,
                warm_failures INTEGER NOT NULL,
                max_pss_mb REAL NOT NULL,
                model_files_bytes INTEGER NOT NULL
            )""".trimIndent(),
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    fun notes(workspace: String): List<Note> = readableDatabase.rawQuery(
        "SELECT id, workspace, event_id, title, body, created_at, updated_at, search_vectors, cluster_vector FROM notes WHERE workspace=? ORDER BY updated_at DESC",
        arrayOf(workspace),
    ).use { cursor -> buildList {
        while (cursor.moveToNext()) add(cursor.toNote())
    } }

    fun note(id: String): Note? = readableDatabase.rawQuery(
        "SELECT id, workspace, event_id, title, body, created_at, updated_at, search_vectors, cluster_vector FROM notes WHERE id=?",
        arrayOf(id),
    ).use { cursor -> if (cursor.moveToFirst()) cursor.toNote() else null }

    fun upsert(input: NoteInput, searchVectors: List<FloatArray>, clusterVector: FloatArray, workspace: String = "personal") {
        val now = System.currentTimeMillis()
        val existing = note(input.id)
        val values = ContentValues().apply {
            put("id", input.id)
            put("workspace", workspace)
            put("event_id", input.eventId)
            put("title", input.title)
            put("body", input.text)
            put("created_at", existing?.createdAt ?: now)
            put("updated_at", now)
            put("search_vectors", FloatCodec.encode(searchVectors))
            put("cluster_vector", FloatCodec.encode(listOf(clusterVector)))
        }
        writableDatabase.insertWithOnConflict("notes", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun delete(id: String) { writableDatabase.delete("notes", "id=?", arrayOf(id)) }

    fun importedSample(id: String): Boolean = readableDatabase.rawQuery(
        "SELECT 1 FROM notes WHERE id=? AND workspace='sample'", arrayOf(id),
    ).use { it.moveToFirst() }

    fun saveFeedback(itemType: String, itemId: String, positive: Boolean) {
        val values = ContentValues().apply {
            put("item_type", itemType)
            put("item_id", itemId)
            put("decision", if (positive) 1 else -1)
            put("created_at", System.currentTimeMillis())
        }
        writableDatabase.insertWithOnConflict("feedback", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun feedback(itemType: String, itemId: String): Int? = readableDatabase.rawQuery(
        "SELECT decision FROM feedback WHERE item_type=? AND item_id=?", arrayOf(itemType, itemId),
    ).use { if (it.moveToFirst()) it.getInt(0) else null }

    fun savePerformance(result: PerformanceResult) {
        val values = ContentValues().apply {
            put("created_at", result.measuredAt)
            put("cold_p95_ms", result.coldP95Ms.takeIf { it.isFinite() } ?: -1.0)
            put("warm_p95_ms", result.warmP95Ms.takeIf { it.isFinite() } ?: -1.0)
            put("cold_failures", result.coldFailures)
            put("warm_failures", result.warmFailures)
            put("max_pss_mb", result.maxAdditionalPssMb)
            put("model_files_bytes", result.modelFilesBytes)
        }
        writableDatabase.insert("performance_runs", null, values)
    }

    fun dailySummaries(days: Int = 7): List<DailyUsageSummary> {
        val since = System.currentTimeMillis() - days * 24L * 60L * 60L * 1000L
        val values = linkedMapOf<String, DailyUsageSummary>()
        readableDatabase.rawQuery(
            """SELECT date(created_at/1000,'unixepoch','localtime'),
                sum(CASE WHEN decision=1 THEN 1 ELSE 0 END),
                sum(CASE WHEN decision=-1 THEN 1 ELSE 0 END)
                FROM feedback WHERE created_at>=? GROUP BY 1""".trimIndent(), arrayOf(since.toString()),
        ).use { cursor ->
            while (cursor.moveToNext()) values[cursor.getString(0)] = DailyUsageSummary(cursor.getString(0), cursor.getInt(1), cursor.getInt(2), null, null, 0)
        }
        readableDatabase.rawQuery(
            """SELECT date(created_at/1000,'unixepoch','localtime'), cold_p95_ms, warm_p95_ms, cold_failures + warm_failures
                FROM performance_runs WHERE created_at>=? ORDER BY created_at DESC""".trimIndent(), arrayOf(since.toString()),
        ).use { cursor ->
            while (cursor.moveToNext()) {
                val day = cursor.getString(0)
                val existing = values[day] ?: DailyUsageSummary(day, 0, 0, null, null, 0)
                if (existing.coldP95Ms == null) values[day] = existing.copy(
                    coldP95Ms = cursor.getDouble(1).takeIf { it >= 0 },
                    warmP95Ms = cursor.getDouble(2).takeIf { it >= 0 },
                    failedRuns = cursor.getInt(3),
                )
            }
        }
        return values.values.sortedByDescending { it.day }
    }

    fun evalEmbedding(docId: String, modelHash: String): Note? = readableDatabase.rawQuery(
        "SELECT search_vectors, cluster_vector FROM eval_embeddings WHERE doc_id=? AND model_hash=?",
        arrayOf(docId, modelHash),
    ).use { cursor ->
        if (!cursor.moveToFirst()) null else Note(
            id = docId,
            workspace = "evaluation",
            eventId = null,
            title = "",
            text = "",
            createdAt = 0,
            updatedAt = 0,
            searchVectors = FloatCodec.decode(cursor.getBlob(0)),
            clusterVector = FloatCodec.decodeOne(cursor.getBlob(1)),
        )
    }

    fun saveEvalEmbedding(docId: String, modelHash: String, searchVectors: List<FloatArray>, clusterVector: FloatArray) {
        val values = ContentValues().apply {
            put("doc_id", docId)
            put("model_hash", modelHash)
            put("search_vectors", FloatCodec.encode(searchVectors))
            put("cluster_vector", FloatCodec.encode(listOf(clusterVector)))
        }
        writableDatabase.insertWithOnConflict("eval_embeddings", null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun android.database.Cursor.toNote() = Note(
        id = getString(0), workspace = getString(1), eventId = if (isNull(2)) null else getString(2),
        title = getString(3), text = getString(4), createdAt = getLong(5), updatedAt = getLong(6),
        searchVectors = FloatCodec.decode(getBlob(7)), clusterVector = FloatCodec.decodeOne(getBlob(8)),
    )
}
