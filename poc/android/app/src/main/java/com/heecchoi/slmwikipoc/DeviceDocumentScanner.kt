package com.heecchoi.slmwikipoc

import android.content.Context
import android.os.Environment
import android.os.storage.StorageManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.file.Files
import java.util.ArrayDeque
import java.util.Locale

data class DeviceScanDiscovery(
    val files: List<File>,
    val unsupported: Int,
    val unreadableDirectories: Int,
    val availableVolumes: Int,
) {
    val complete: Boolean get() = unreadableDirectories == 0
}

object DeviceDocumentScanner {
    suspend fun discover(context: Context): DeviceScanDiscovery = withContext(Dispatchers.IO) {
        val storage = context.getSystemService(StorageManager::class.java)
        val files = mutableListOf<File>()
        val visited = HashSet<String>()
        var unsupported = 0
        var unreadable = 0
        var availableVolumes = 0
        storage.storageVolumes.forEach { volume ->
            currentCoroutineContext().ensureActive()
            val root = volume.directory
            if (root == null || volume.state !in setOf(Environment.MEDIA_MOUNTED, Environment.MEDIA_MOUNTED_READ_ONLY) || !Environment.isExternalStorageManager(root)) {
                unreadable++
                return@forEach
            }
            availableVolumes++
            val pending = ArrayDeque<File>()
            pending.add(root)
            while (pending.isNotEmpty()) {
                currentCoroutineContext().ensureActive()
                val entry = pending.removeLast()
                if (try { Files.isSymbolicLink(entry.toPath()) } catch (_: Exception) { unreadable++; true }) continue
                val path = try { entry.canonicalPath } catch (_: Exception) { unreadable++; continue }
                if (!visited.add(path)) continue
                if (entry.isDirectory) {
                    if (entry.parentFile?.name == "Android" && entry.name in setOf("data", "obb")) continue
                    val children = try { entry.listFiles() } catch (_: SecurityException) { null }
                    if (children == null) unreadable++ else children.forEach(pending::add)
                } else if (entry.isFile) {
                    if (entry.extension.lowercase(Locale.ROOT) in setOf("txt", "md", "pdf")) files += entry
                    else unsupported++
                } else if (entry.exists()) {
                    unreadable++
                }
            }
        }
        DeviceScanDiscovery(files.sortedBy(File::getAbsolutePath), unsupported, unreadable, availableVolumes)
    }
}
