package com.heecchoi.slmwikipoc

import android.content.ComponentName
import android.content.Context
import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class LifeNotificationIntegrationTest {
    @Test fun collectionIsDisabledWithoutDeletingLegacyRecords() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val suffix = UUID.randomUUID().toString()
        val dbName = "legacy-test-$suffix.db"
        val prefs = "legacy-test-$suffix"
        val store = LocalStore(context, dbName)
        try {
            val repo = LifeRepository(context, store, prefs) { }
            repo.setAllowed("example.shop", true)
            repo.setCollectionEnabled(true)
            val now = System.currentTimeMillis()
            assertTrue(repo.recordNotification("example.shop", "상점", "old-key", "기존 알림", "기존 내용", now, "UTC", now))
            stopLegacyNotificationCollection(context, prefs)
            assertFalse(repo.collectionEnabled())
            assertEquals("기존 알림", repo.records().single().title)
            @Suppress("DEPRECATION")
            val info = context.packageManager.getServiceInfo(ComponentName(context, LifeNotificationListener::class.java),
                PackageManager.MATCH_DISABLED_COMPONENTS)
            assertFalse(info.enabled)
        } finally {
            store.close()
            context.deleteDatabase(dbName)
            context.getSharedPreferences(prefs, Context.MODE_PRIVATE).edit().clear().commit()
        }
    }
}
