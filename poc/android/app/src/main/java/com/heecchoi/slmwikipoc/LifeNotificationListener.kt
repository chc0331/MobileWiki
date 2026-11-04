package com.heecchoi.slmwikipoc

import android.app.Notification
import android.content.ComponentName
import android.content.Context
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.time.ZoneId
import java.util.concurrent.Executors

class LifeNotificationListener : NotificationListenerService() {
    private val dispatcher = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val scope = CoroutineScope(SupervisorJob() + dispatcher)

    companion object {
        @Volatile private var current: LifeNotificationListener? = null
        fun captureCurrent(context: Context) {
            val service = current
            if (service != null) service.snapshot()
            else runCatching { requestRebind(ComponentName(context, LifeNotificationListener::class.java)) }
        }
        fun activePackages(): Set<String> = runCatching {
            current?.activeNotifications?.map { it.packageName }?.toSet().orEmpty()
        }.getOrDefault(emptySet())
    }

    override fun onListenerConnected() {
        super.onListenerConnected()
        current = this
        LifeSignals.connected.value = true
        snapshot()
    }

    override fun onListenerDisconnected() {
        current = null
        LifeSignals.connected.value = false
        super.onListenerDisconnected()
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) {
        val repo = LifeServices.repository(this)
        if (!repo.acceptsPackage(sbn.packageName)) return
        scope.launch { captureSafely(repo, sbn) }
    }

    override fun onNotificationRemoved(sbn: StatusBarNotification, rankingMap: RankingMap, reason: Int) {
        scope.launch {
            LifeServices.repository(this@LifeNotificationListener).notificationRemoved(
                sbn.key, android.os.Build.VERSION.SDK_INT >= 34 && reason == REASON_LOCKDOWN,
            )
        }
    }

    private fun snapshot() {
        val repo = LifeServices.repository(this)
        if (!repo.collectionEnabled()) return
        val allActive = runCatching { activeNotifications?.toList().orEmpty() }.getOrElse {
            LifeSignals.connected.value = false
            return
        }
        val active = allActive.filter { repo.acceptsPackage(it.packageName) }
        scope.launch {
            repo.reconcileActive(active.map { it.key }.toSet(), allActive.map { it.key }.toSet())
            active.forEach { captureSafely(repo, it) }
        }
    }

    private fun captureSafely(repo: LifeRepository, sbn: StatusBarNotification) {
        try {
            capture(repo, sbn)
        } catch (_: Exception) {
            if (repo.acceptsPackage(sbn.packageName)) repo.recordExcluded()
        }
    }

    private fun capture(repo: LifeRepository, sbn: StatusBarNotification) {
        if (!repo.acceptsPackage(sbn.packageName)) return
        val extras = sbn.notification.extras ?: return
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()
        val parts = mutableListOf<String>()
        fun add(value: CharSequence?) {
            val text = value?.toString()?.trim().orEmpty()
            if (text.isNotBlank() && parts.none { it == text }) parts += text
        }
        val big = extras.getCharSequence(Notification.EXTRA_BIG_TEXT)
        add(big ?: extras.getCharSequence(Notification.EXTRA_TEXT))
        extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)?.forEach(::add)
        @Suppress("DEPRECATION")
        extras.getParcelableArray(Notification.EXTRA_MESSAGES)?.let { messages ->
            Notification.MessagingStyle.Message.getMessagesFromBundleArray(messages).forEach { add(it.text) }
        }
        val label = runCatching {
            packageManager.getApplicationLabel(packageManager.getApplicationInfo(sbn.packageName, 0)).toString()
        }.getOrDefault(sbn.packageName)
        repo.recordNotification(sbn.packageName, label, sbn.key, title, parts.joinToString("\n"), sbn.postTime, ZoneId.systemDefault().id)
    }

    override fun onDestroy() {
        current = null
        LifeSignals.connected.value = false
        scope.cancel()
        dispatcher.close()
        super.onDestroy()
    }
}
