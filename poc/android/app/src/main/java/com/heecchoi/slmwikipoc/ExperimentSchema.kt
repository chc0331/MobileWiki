package com.heecchoi.slmwikipoc

import android.database.sqlite.SQLiteDatabase

object ExperimentSchema {
    fun create(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE experiment_inputs (
            id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
            ground_truth TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
            source_name TEXT, source_mime TEXT, source_path TEXT, page_texts TEXT,
            device_path TEXT UNIQUE, device_size INTEGER, device_modified_at INTEGER,
            cluster_vector BLOB, model_hash TEXT, pipeline_version TEXT, scan_error TEXT
        )""")
        db.execSQL("""CREATE TABLE experiment_run (
            id INTEGER PRIMARY KEY CHECK(id=1), signature TEXT NOT NULL, model_hash TEXT NOT NULL,
            threshold REAL NOT NULL, created_at INTEGER NOT NULL, embedding_ms INTEGER NOT NULL,
            clustering_ms INTEGER NOT NULL, computed INTEGER NOT NULL, reused INTEGER NOT NULL,
            labeled_count INTEGER NOT NULL, labeled_pairs INTEGER NOT NULL, false_merges INTEGER NOT NULL,
            precision REAL, recall REAL, f1 REAL
        )""")
        db.execSQL("""CREATE TABLE experiment_run_items (
            run_id INTEGER NOT NULL REFERENCES experiment_run(id) ON DELETE CASCADE,
            input_id TEXT NOT NULL REFERENCES experiment_inputs(id) ON DELETE CASCADE,
            group_id INTEGER NOT NULL, group_size INTEGER NOT NULL, x REAL NOT NULL, y REAL NOT NULL,
            PRIMARY KEY(run_id,input_id)
        )""")
    }
}
