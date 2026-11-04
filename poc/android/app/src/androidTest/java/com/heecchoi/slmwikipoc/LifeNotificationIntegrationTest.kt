package com.heecchoi.slmwikipoc

import android.app.NotificationManager
import android.content.ComponentName
import android.os.Build
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.FileInputStream
import java.util.UUID

@RunWith(AndroidJUnit4::class)
class LifeNotificationIntegrationTest {
    @Test fun listenerFiltersAndImportsCurrentThenUpdatedNotification() {
        assumeTrue(Build.FINGERPRINT.contains("generic") || Build.MODEL.contains("sdk_gphone"))
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val context = instrumentation.targetContext
        val repo = LifeServices.repository(context)
        val originalEnabled = repo.collectionEnabled()
        val shellWasAllowed = "com.android.shell" in repo.allowedPackages()
        val component = context.packageName + "/.LifeNotificationListener"
        val listenerWasAllowed = context.getSystemService(NotificationManager::class.java)
            .isNotificationListenerAccessGranted(ComponentName(context, LifeNotificationListener::class.java))
        val marker = "LifePoc-" + UUID.randomUUID().toString()
        fun shell(command: String) {
            instrumentation.uiAutomation.executeShellCommand(command).use { pfd ->
                FileInputStream(pfd.fileDescriptor).bufferedReader().use { it.readText() }
            }
        }
        fun await(stage: String, condition: () -> Boolean) {
            val deadline = System.currentTimeMillis() + 20_000
            while (!condition() && System.currentTimeMillis() < deadline) Thread.sleep(200)
            val diagnostics = repo.records().filter { it.title == marker }.map { record ->
                record.id to repo.versions(record.id).map { it.text }
            }
            assertTrue("알림 리스너 $stage 시간 초과: $diagnostics", condition())
        }
        try {
            repo.setCollectionEnabled(false)
            repo.setAllowed("com.android.shell", false)
            shell("cmd notification allow_listener " + component)
            await("연결") { LifeSignals.connected.value }
            shell("cmd notification post -t " + marker + " " + marker + " order-prepared")
            Thread.sleep(500)
            assertTrue(repo.records().none { it.title == marker })
            repo.setAllowed("com.android.shell", true)
            repo.setCollectionEnabled(true)
            LifeNotificationListener.captureCurrent(context)
            await("최초 수집") { repo.records().any { it.title == marker } }
            val id = repo.records().first { it.title == marker }.id
            shell("cmd notification post -t " + marker + " " + marker + " order-delivered")
            await("갱신") { repo.versions(id).size >= 2 }
            assertEquals(listOf("order-prepared", "order-delivered"), repo.versions(id).map { it.text })
            LifeNotificationListener.captureCurrent(context)
            Thread.sleep(500)
            assertEquals(2, repo.versions(id).size)
        } finally {
            repo.setCollectionEnabled(false)
            repo.records().filter { it.title == marker }.forEach { repo.deleteRecord(it.id) }
            repo.setAllowed("com.android.shell", shellWasAllowed)
            repo.setCollectionEnabled(originalEnabled)
            if (!listenerWasAllowed) shell("cmd notification disallow_listener " + component)
        }
    }
}
