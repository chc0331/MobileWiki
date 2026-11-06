package com.heecchoi.slmwikipoc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.Random

class LargeClusteringTest {
    @Test fun optimizedAverageLinkageMatchesExistingThresholdGroups() {
        val random = Random(72)
        repeat(20) {
            val notes = (0 until 24).map { index ->
                val vector = FloatArray(8) { random.nextFloat() * 2 - 1 }
                Note(
                    id = "note-$index", workspace = "experiment", eventId = null,
                    title = "문서 $index", text = "내용", createdAt = 0, updatedAt = 0,
                    searchVectors = listOf(vector), clusterVector = vector,
                )
            }
            listOf(0.0, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0).forEach { threshold ->
                val expected = AverageLinkageGroups.cluster(notes, threshold).map { group -> group.notes.map(Note::id).toSet() }.toSet()
                val actual = AverageLinkageGroups.clusterLarge(notes, threshold).map { group -> group.notes.map(Note::id).toSet() }.toSet()
                assertEquals("threshold=$threshold", expected, actual)
            }
        }
    }

    @Test fun clustersEveryDocumentInALargerSet() {
        val random = Random(91)
        val notes = (0 until 500).map { index ->
            val vector = FloatArray(8) { random.nextFloat() * 2 - 1 }
            Note("note-$index", "experiment", null, "문서 $index", "내용", 0, 0, listOf(vector), vector)
        }
        val groups = AverageLinkageGroups.clusterLarge(notes, 0.35)
        assertEquals(notes.map(Note::id).toSet(), groups.flatMap { group -> group.notes.map(Note::id) }.toSet())
        assertEquals(notes.size, groups.sumOf { it.notes.size })
    }

    @Test fun oversizedExactClusteringReportsFailureBeforeAllocatingDistanceMatrix() {
        val vector = floatArrayOf(1f)
        val base = Note("0", "experiment", null, "문서", "내용", 0, 0, listOf(vector), vector)
        val notes = List(65_537) { base.copy(id = it.toString()) }
        val result = runCatching { AverageLinkageGroups.clusterLarge(notes, 0.05) }
        assertTrue(result.exceptionOrNull() is IllegalStateException)
        assertTrue(result.exceptionOrNull()?.message?.contains("메모리가 부족") == true)
    }
}
