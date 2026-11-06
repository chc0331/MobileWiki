package com.heecchoi.slmwikipoc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ExperimentDomainTest {
    private fun item(id: String, label: String?) = ExperimentItem(id, "text", id, "내용", label,
        0, 0, 0, null, floatArrayOf(1f, 0f), "model", ExperimentStore.PIPELINE_VERSION)

    @Test fun chunkRangesCoverEveryBodyTokenEvenWithLongPrefixes() {
        for (prefix in listOf(10, 89, 150, 300, 445)) {
            for (length in listOf(1, 50, 420, 421, 1_000, 5_000)) {
                val ranges = BodyChunking.ranges(length, prefix)
                val covered = ranges.flatMap { it.toList() }.toSet()
                assertEquals("prefix=$prefix length=$length", (0 until length).toSet(), covered)
                assertTrue(ranges.all { it.last - it.first + 1 + prefix + 2 <= 511 })
            }
        }
        assertEquals(listOf(0..419, 356..775, 712..999), BodyChunking.ranges(1000, 40))
    }

    @Test(expected = IllegalArgumentException::class)
    fun tooLongTitleIsRejectedInsteadOfDroppingBody() {
        BodyChunking.ranges(1000, 500)
    }

    @Test fun qualityUsesOnlyLabeledRecordsAndAllGroupsMustBePresent() {
        val a = item("a", "여행")
        val b = item("b", "여행")
        val c = item("c", "업무")
        val d = item("d", null)
        val groups = listOf(NoteGroup(0, "", listOf(a.note(), b.note(), c.note())), NoteGroup(1, "", listOf(d.note())))
        val q = ExperimentQualityCalculator.calculate(listOf(a, b, c, d), groups)
        assertEquals(3, q.labeled)
        assertEquals(3L, q.pairs)
        assertEquals(2L, q.falseMerges)
        assertEquals(1.0 / 3.0, q.precision!!, 1e-9)
        assertEquals(1.0, q.recall!!, 1e-9)
        assertEquals(0.5, q.f1!!, 1e-9)
    }

    @Test(expected = IllegalArgumentException::class)
    fun qualityRejectsMissingUnlabeledInputs() {
        val a = item("a", "주제")
        val b = item("b", null)
        ExperimentQualityCalculator.calculate(listOf(a, b), listOf(NoteGroup(0, "", listOf(a.note()))))
    }

    @Test fun signatureChangesOnlyForDatasetRevisionModelOrThreshold() {
        val a = item("a", null)
        val b = item("b", "주제")
        val baseline = ExperimentStore.signature(listOf(a, b), 0.05, "model")
        assertEquals(baseline, ExperimentStore.signature(listOf(b, a), 0.05, "model"))
        assertFalse(baseline == ExperimentStore.signature(listOf(a.copy(revision = 1), b), 0.05, "model"))
        assertFalse(baseline == ExperimentStore.signature(listOf(a, b), 0.06, "model"))
        assertFalse(baseline == ExperimentStore.signature(listOf(a, b), 0.05, "other"))
    }
}
