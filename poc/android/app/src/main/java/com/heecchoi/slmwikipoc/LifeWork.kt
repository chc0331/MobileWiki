package com.heecchoi.slmwikipoc

import android.content.Context
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
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
        while (true) {
            currentCoroutineContext().ensureActive()
            val item = repo.nextPending() ?: break
            try {
                val model = LifeServices.model(applicationContext)
                model.load()
                val vectors = model.embedNote(item.title, item.body)
                currentCoroutineContext().ensureActive()
                repo.completePending(item, vectors)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                if (repo.failPending(item, error.javaClass.simpleName)) return@withLock Result.retry()
            } catch (error: OutOfMemoryError) {
                if (repo.failPending(item, "메모리 부족")) return@withLock Result.retry()
            }
        }
        LifeSignals.changed()
        Result.success()
    } }
}
