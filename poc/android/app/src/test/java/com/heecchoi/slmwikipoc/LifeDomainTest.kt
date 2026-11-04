package com.heecchoi.slmwikipoc

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.time.LocalDate

class LifeDomainTest {
    private fun record(id: String, key: String?, vector: FloatArray, truth: String? = null) =
        LifeRecord(id, "memo", id, id, 0L, 0L, null, null, null, key, listOf(vector), vector, "ready", null, truth)

    @Test fun relativeDatesUseKnownSourceDateAndNeverGuessMissingYear() {
        val base = LocalDate.of(2026, 12, 31)
        assertEquals(LocalDate.of(2027, 1, 1), LifeRules.date("내일 배송 예정", base))
        assertEquals(LocalDate.of(2027, 1, 2), LifeRules.date("모레 배송 예정", base))
        assertEquals(LocalDate.of(2026, 11, 20), LifeRules.date("2026년 11월 20일까지 반품", base))
        assertNull(LifeRules.date("11월 20일까지 반품", base))
        assertNull(LifeRules.date("내일 배송", null))
        assertNull(LifeRules.date("2026-02-30", base))
    }

    @Test fun deliveryIsEventWhileExplicitReturnIsTask() {
        val proposals = LifeRules.proposals("내일 배송 예정\n2026-11-20까지 반품 신청", LocalDate.of(2026, 11, 5))
        assertEquals("event", proposals[0].first)
        assertEquals(LocalDate.of(2026, 11, 6), proposals[0].third)
        assertEquals("task", proposals[1].first)
        assertEquals(LocalDate.of(2026, 11, 20), proposals[1].third)
        assertTrue(LifeRules.proposals("배송이 완료됐습니다", LocalDate.now()).isEmpty())
        assertTrue(LifeRules.proposals("반품할 필요 없음\n결제하지 마세요\n배송 예정이 취소되었습니다", LocalDate.now()).isEmpty())
    }

    @Test fun orderKeyNeedsUnambiguousNumberAndMerchant() {
        val text = "상점: 가람상점\n주문번호: A12345"
        assertEquals("가람상점|A12345", LifeRules.orderKey("", text, null))
        assertNull(LifeRules.orderKey("", "주문번호: A12345", null))
        assertNull(LifeRules.orderKey("", text + "\n주문번호: B12345", null))
        assertNull(LifeRules.orderKey("", text + "\n상점: 나무상점", null))
        assertFalse(LifeRules.orderKey("", text, null) == LifeRules.orderKey("", text.replace("가람", "나무"), null))
    }

    @Test fun hybridKeepsConflictingOrdersSeparateAndLeavesAmbiguityUnassigned() {
        val vector = floatArrayOf(1f, 0f)
        val records = listOf(
            record("a", "상점|A100", vector, "A"),
            record("b", "상점|B200", vector, "B"),
            record("unknown", null, vector, "A"),
        )
        val model = LifeGrouping.model(records, 0.05)
        val hybrid = LifeGrouping.hybrid(records, model, 0.05)
        assertEquals(setOf(setOf("a"), setOf("b"), setOf("unknown")), hybrid.map { it.toSet() }.toSet())
        assertEquals(1L, LifeGrouping.metrics(records, model).positivePairs)
        assertEquals(2L, LifeGrouping.metrics(records, model).falsePositive)
        assertEquals(0L, LifeGrouping.metrics(records, hybrid).falsePositive)
    }

    @Test fun hybridCanJoinOneUnambiguousModelNeighbourAndMatchExplicitIdAcrossClusters() {
        val records = listOf(
            record("a", "상점|A100", floatArrayOf(1f, 0f), "A"),
            record("b", "상점|A100", floatArrayOf(0f, 1f), "A"),
            record("c", null, floatArrayOf(1f, 0f), "A"),
        )
        val model = LifeGrouping.model(records, 0.05)
        val hybrid = LifeGrouping.hybrid(records, model, 0.05).map { it.toSet() }.toSet()
        // 기록 c는 같은 주문의 모든 확인된 단서와 가깝지 않으므로 자동 연결하지 않는다.
        assertEquals(setOf(setOf("a", "b"), setOf("c")), hybrid)
        val simple = records.filter { it.id != "b" }
        val simpleGroups = LifeGrouping.hybrid(simple, LifeGrouping.model(simple, 0.05), 0.05)
        assertEquals(setOf(setOf("a", "c")), simpleGroups.map { it.toSet() }.toSet())
    }

    @Test(expected = IllegalArgumentException::class)
    fun metricsRejectPartialResultsInsteadOfComparingMissingIdsAsEqual() {
        val records = listOf(record("a", null, floatArrayOf(1f), "A"), record("b", null, floatArrayOf(1f), "B"))
        LifeGrouping.metrics(records, listOf(listOf("a")))
    }

    @Test(expected = IllegalArgumentException::class)
    fun metricsRejectMissingUnlabeledRecords() {
        val records = listOf(record("a", null, floatArrayOf(1f), "A"), record("b", null, floatArrayOf(1f)))
        LifeGrouping.metrics(records, listOf(listOf("a")))
    }

    @Test fun syntheticSplitsHaveEnoughPairsAndDistinctOrders() {
        val dev = LifeSynthetic.items(LifeSynthetic.DEV)
        val test = LifeSynthetic.items(LifeSynthetic.TEST)
        assertTrue(dev.map { it.orderId }.toSet().intersect(test.map { it.orderId }.toSet()).isEmpty())
        val records = test.map { item ->
            record(item.id, LifeRules.orderKey(item.title, item.text, item.merchant), floatArrayOf(1f, 0f), item.orderId)
        }
        val metrics = LifeGrouping.metrics(records, records.map { listOf(it.id) })
        assertTrue(metrics.positivePairs >= 50)
        assertTrue(metrics.negativePairs >= 50)
    }
}
