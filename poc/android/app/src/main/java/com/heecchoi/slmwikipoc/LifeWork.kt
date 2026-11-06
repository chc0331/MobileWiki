package com.heecchoi.slmwikipoc

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.TimeUnit

object LifeWork {
    val processingMutex = Mutex()
    fun enqueue(context: Context) {
        WorkManager.getInstance(context.applicationContext).enqueueUniqueWork(
            "life-processing", ExistingWorkPolicy.APPEND_OR_REPLACE, OneTimeWorkRequestBuilder<LifeWorker>().build(),
        )
    }

    fun scheduleCleanup(context: Context) {
        WorkManager.getInstance(context.applicationContext).enqueueUniquePeriodicWork(
            "life-expiry", ExistingPeriodicWorkPolicy.KEEP,
            PeriodicWorkRequestBuilder<LifeWorker>(12, TimeUnit.HOURS).build(),
        )
    }
}

class LifeWorker(context: Context, params: WorkerParameters) : CoroutineWorker(context, params) {
    override suspend fun doWork(): Result = withContext(Dispatchers.IO) { LifeWork.processingMutex.withLock {
        val repo = LifeServices.repository(applicationContext)
        repo.cleanupExpired()
        // 숨긴 생활 기능의 기존 보관 기간은 유지하되, 백그라운드 임베딩은 실행하지 않는다.
        LifeSignals.changed()
        Result.success()
    } }
}
