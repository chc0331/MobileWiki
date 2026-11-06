package com.heecchoi.slmwikipoc

import android.app.Application
import android.content.Context

class PocApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        stopLegacyNotificationCollection(this)
        LifeWork.scheduleCleanup(this)
    }
}

fun stopLegacyNotificationCollection(context: Context, preferences: String = "life-poc") {
    val prefs = context.getSharedPreferences(preferences, Context.MODE_PRIVATE)
    if (prefs.getBoolean("enabled", false)) prefs.edit().putBoolean("enabled", false).commit()
}
