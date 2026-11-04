package com.heecchoi.slmwikipoc

import android.database.sqlite.SQLiteDatabase

object LifeSchema {
    fun create(db: SQLiteDatabase) {
        db.execSQL("""CREATE TABLE life_records (
            id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL DEFAULT '',
            body TEXT NOT NULL DEFAULT '', note_id TEXT UNIQUE, note_updated_at INTEGER, package_name TEXT, source_name TEXT,
            notification_key TEXT, active INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL, source_time INTEGER, zone_id TEXT,
            revision INTEGER NOT NULL DEFAULT 0, cluster_vector BLOB, search_vectors BLOB,
            processing TEXT NOT NULL DEFAULT 'pending', error TEXT, attempts INTEGER NOT NULL DEFAULT 0, truth_order TEXT
        )""")
        db.execSQL("CREATE INDEX life_records_notification ON life_records(package_name, notification_key, active)")
        db.execSQL("""CREATE TABLE life_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL REFERENCES life_records(id) ON DELETE CASCADE,
            title TEXT NOT NULL, body TEXT NOT NULL, digest TEXT NOT NULL, captured_at INTEGER NOT NULL,
            posted_at INTEGER NOT NULL, zone_id TEXT NOT NULL, cluster_vector BLOB, search_vectors BLOB,
            processing TEXT NOT NULL DEFAULT 'pending', error TEXT, attempts INTEGER NOT NULL DEFAULT 0
        )""")
        db.execSQL("CREATE INDEX life_versions_expiry ON life_versions(captured_at)")
        db.execSQL("""CREATE TABLE life_tombstones (
            notification_key TEXT NOT NULL, digest TEXT NOT NULL, expires_at INTEGER NOT NULL,
            PRIMARY KEY(notification_key, digest)
        )""")
        db.execSQL("""CREATE TABLE life_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL REFERENCES life_records(id) ON DELETE CASCADE,
            version_id INTEGER REFERENCES life_versions(id) ON DELETE CASCADE, kind TEXT NOT NULL,
            title TEXT NOT NULL, due_date TEXT, evidence TEXT NOT NULL, fingerprint TEXT NOT NULL UNIQUE,
            state TEXT NOT NULL DEFAULT 'pending'
        )""")
        db.execSQL("""CREATE TABLE life_tasks (
            id TEXT PRIMARY KEY NOT NULL, source_id TEXT REFERENCES life_records(id) ON DELETE SET NULL,
            source_version_id INTEGER REFERENCES life_versions(id) ON DELETE SET NULL,
            source_kind TEXT, kind TEXT NOT NULL, title TEXT NOT NULL, due_date TEXT, done INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL
        )""")
        db.execSQL("""CREATE TABLE life_groups (
            id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
            created_at INTEGER NOT NULL
        )""")
        db.execSQL("""CREATE TABLE life_memberships (
            group_id TEXT NOT NULL REFERENCES life_groups(id) ON DELETE CASCADE,
            record_id TEXT NOT NULL REFERENCES life_records(id) ON DELETE CASCADE,
            membership TEXT NOT NULL, PRIMARY KEY(group_id, record_id)
        )""")
        db.execSQL("CREATE INDEX life_members_record ON life_memberships(record_id)")
        db.execSQL("""CREATE TABLE life_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, threshold REAL NOT NULL,
            data_hash TEXT NOT NULL, model_hash TEXT NOT NULL, model_groups TEXT NOT NULL, hybrid_groups TEXT NOT NULL
        )""")
        db.execSQL("""CREATE TABLE life_reviews (
            dataset TEXT NOT NULL, item_id TEXT NOT NULL, reviewed INTEGER NOT NULL,
            PRIMARY KEY(dataset, item_id)
        )""")
        db.execSQL("""CREATE TABLE life_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL, dataset TEXT NOT NULL,
            model_hash TEXT NOT NULL, threshold REAL NOT NULL, positives INTEGER NOT NULL, negatives INTEGER NOT NULL,
            model_precision REAL NOT NULL, model_recall REAL NOT NULL, model_f1 REAL NOT NULL, model_false_merges INTEGER NOT NULL,
            hybrid_precision REAL NOT NULL, hybrid_recall REAL NOT NULL, hybrid_f1 REAL NOT NULL, hybrid_false_merges INTEGER NOT NULL
        )""")
        db.execSQL("""CREATE TABLE life_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at INTEGER NOT NULL,
            scenario TEXT NOT NULL, method TEXT NOT NULL, success INTEGER NOT NULL,
            seconds INTEGER, issue TEXT NOT NULL DEFAULT ''
        )""")
    }
}
