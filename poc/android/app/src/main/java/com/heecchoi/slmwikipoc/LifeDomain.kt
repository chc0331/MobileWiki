package com.heecchoi.slmwikipoc

import java.time.LocalDate
import java.time.DateTimeException
import java.util.Locale

data class LifeRecord(
    val id: String,
    val kind: String,
    val title: String,
    val text: String,
    val createdAt: Long,
    val updatedAt: Long,
    val sourceName: String?,
    val noteId: String?,
    val packageName: String?,
    val orderKey: String?,
    val searchVectors: List<FloatArray>,
    val clusterVector: FloatArray?,
    val processing: String,
    val error: String?,
    val truthOrder: String?,
)

data class LifeVersion(
    val id: Long,
    val recordId: String,
    val title: String,
    val text: String,
    val capturedAt: Long,
    val postedAt: Long,
    val zoneId: String,
    val processing: String,
)

data class LifeCandidate(
    val id: Long,
    val recordId: String,
    val versionId: Long?,
    val kind: String,
    val title: String,
    val dueDate: String?,
    val evidence: String,
)

data class LifeTask(
    val id: String,
    val sourceId: String?,
    val kind: String,
    val title: String,
    val dueDate: String?,
    val done: Boolean,
    val sourceExpired: Boolean,
)

data class LifeManualGroup(val id: String, val kind: String, val title: String, val members: List<Pair<String, String>>)
data class LifeEvaluationResult(
    val dataset: String, val threshold: Double, val positives: Long, val negatives: Long,
    val modelPrecision: Double, val modelRecall: Double, val modelF1: Double, val modelFalseMerges: Long,
    val hybridPrecision: Double, val hybridRecall: Double, val hybridF1: Double, val hybridFalseMerges: Long,
    val modelHash: String, val createdAt: Long,
)

data class LifePairMetrics(
    val labeledRecords: Int,
    val positivePairs: Long,
    val negativePairs: Long,
    val truePositive: Long,
    val falsePositive: Long,
    val falseNegative: Long,
    val precision: Double?,
    val recall: Double?,
    val f1: Double?,
)

object LifeRules {
    private val orderPattern = Regex("""주문[ \t]*(?:번호|#|no\.?)[ \t]*[:：#]?[ \t]*([A-Za-z0-9][A-Za-z0-9-]{3,39})""", RegexOption.IGNORE_CASE)
    private val merchantPattern = Regex("""(?:상점|판매처|쇼핑몰)[ \t]*[:：][ \t]*([^\n,|]{1,40})""")
    private val fullDate = Regex("""(?<!\d)(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?""")
    private val monthDay = Regex("""(?<!\d)(\d{1,2})\s*(?:월|/)\s*(\d{1,2})\s*일?""")
    private val taskWords = Regex("""반품|교환|결제|구매\s*확정|수령|신청|제출|준비""")
    private val eventWords = Regex("""배송\s*예정|도착\s*예정|배송\s*도착""")
    private val negation = Regex("""필요\s*없|불필요|하지\s*(?:않|마)|안\s*(?:해|하)|취소(?:됐|되었|되었습니다|됨)""")

    fun orderKey(title: String, text: String, sourceName: String?): String? {
        val input = title + "\n" + text
        val numbers = orderNumbers(input)
        val number = numbers.singleOrNull() ?: return null
        val merchants = merchantPattern.findAll(input).map { it.groupValues[1].trim() }.distinct().toList()
        if (merchants.size > 1) return null
        val merchant = merchants.singleOrNull() ?: sourceName?.takeIf { it.isNotBlank() } ?: return null
        val normalized = merchant.lowercase(Locale.ROOT).replace(Regex("""[\s\p{Punct}]+"""), "")
        return if (normalized.isBlank()) null else normalized + "|" + number
    }

    private fun orderNumbers(input: String): List<String> =
        orderPattern.findAll(input).map { it.groupValues[1].uppercase(Locale.ROOT) }.distinct().toList()

    fun hasMultipleOrders(title: String, text: String): Boolean = orderNumbers(title + "\n" + text).size > 1

    fun date(text: String, base: LocalDate?): LocalDate? {
        fullDate.find(text)?.let { match ->
            return makeDate(match.groupValues[1].toInt(), match.groupValues[2].toInt(), match.groupValues[3].toInt())
        }
        if (monthDay.containsMatchIn(text)) return null // 연도가 없는 절대 날짜는 사용자가 확인한다.
        if (base == null) return null
        return when {
            "모레" in text -> base.plusDays(2)
            "내일" in text -> base.plusDays(1)
            "오늘" in text -> base
            else -> null
        }
    }

    private fun makeDate(year: Int, month: Int, day: Int): LocalDate? = try {
        LocalDate.of(year, month, day)
    } catch (_: DateTimeException) {
        null
    }

    fun proposals(text: String, base: LocalDate?): List<Triple<String, String, LocalDate?>> {
        val results = mutableListOf<Triple<String, String, LocalDate?>>()
        text.split('\n', '!', '?').map(String::trim).filter(String::isNotBlank).forEach { sentence ->
            if (negation.containsMatchIn(sentence)) return@forEach
            val kind = when {
                taskWords.containsMatchIn(sentence) -> "task"
                eventWords.containsMatchIn(sentence) -> "event"
                else -> return@forEach
            }
            results += Triple(kind, sentence, date(sentence, base))
        }
        return results.distinct()
    }
}

object LifeGrouping {
    fun model(records: List<LifeRecord>, threshold: Double, checkCancelled: () -> Unit = {}): List<List<String>> {
        require(records.all { it.clusterVector != null }) { "임베딩이 끝나지 않은 기록이 있습니다." }
        val notes = records.map { record ->
            val vector = requireNotNull(record.clusterVector)
            Note(record.id, "life", null, record.title, record.text, record.createdAt, record.updatedAt,
                listOf(vector), vector)
        }
        return AverageLinkageGroups.clusterLarge(notes, threshold, checkCancelled).map { group -> group.notes.map(Note::id) }
    }

    // 주문번호가 같은 기록을 먼저 합치고, 번호가 없는 기록은 유사한 주문이 하나뿐일 때만 연결한다.
    fun hybrid(records: List<LifeRecord>, modelGroups: List<List<String>>, threshold: Double): List<List<String>> {
        val keyed = records.filter { it.orderKey != null }.groupBy { requireNotNull(it.orderKey) }
            .mapValues { (_, group) -> group.map(LifeRecord::id).toMutableList() }.toMutableMap()
        val anchors = keyed.mapValues { it.value.toList() }
        val byId = records.associateBy(LifeRecord::id)
        val modelById = modelGroups.flatMap { group -> group.map { it to group } }.toMap()
        val unmatched = mutableListOf<List<String>>()
        records.filter { it.orderKey == null }.forEach { record ->
            if (LifeRules.hasMultipleOrders(record.title, record.text)) {
                unmatched += listOf(record.id)
                return@forEach
            }
            val vector = record.clusterVector
            val modelPeers = modelById[record.id].orEmpty().toSet()
            val matches = if (vector == null) emptyList() else anchors.filter { (_, ids) ->
                ids.any { it in modelPeers } && ids.mapNotNull(byId::get).all { member ->
                    member.clusterVector != null && 1.0 - NoteSearch.cosine(vector, member.clusterVector) <= threshold
                }
            }.keys.toList()
            if (matches.size == 1) keyed.getValue(matches.single()).add(record.id)
            else unmatched += listOf(record.id)
        }
        return keyed.values.map { it.toList() } + unmatched
    }

    fun candidates(record: LifeRecord, groups: List<List<String>>, records: List<LifeRecord>, threshold: Double, checkOrderConflict: Boolean = true): List<List<String>> {
        val vector = record.clusterVector ?: return emptyList()
        val byId = records.associateBy(LifeRecord::id)
        return groups.filter { group ->
            if (record.id in group) return@filter false
            val members = group.mapNotNull(byId::get)
            if (members.isEmpty() || members.any { it.clusterVector == null }) return@filter false
            val keys = members.mapNotNull(LifeRecord::orderKey).toSet()
            if (checkOrderConflict && record.orderKey != null && keys.any { key ->
                    key.substringBefore('|') == record.orderKey.substringBefore('|') && key != record.orderKey
                }) return@filter false
            val distance = members.map { 1.0 - NoteSearch.cosine(vector, requireNotNull(it.clusterVector)) }.average()
            distance <= threshold
        }
    }

    fun metrics(records: List<LifeRecord>, groups: List<List<String>>): LifePairMetrics {
        val labeled = records.filter { !it.truthOrder.isNullOrBlank() }
        val predicted = HashMap<String, Int>()
        groups.forEachIndexed { index, ids -> ids.forEach { id -> check(predicted.put(id, index) == null) { "중복된 군집 구성원입니다." } } }
        require(predicted.keys == records.map { it.id }.toSet() && records.map { it.id }.distinct().size == records.size) {
            "평가 결과에서 기록이 빠졌거나 잘못 포함되었습니다."
        }
        var positive = 0L
        var negative = 0L
        var tp = 0L
        var fp = 0L
        var fn = 0L
        for (i in 0 until labeled.size) for (j in i + 1 until labeled.size) {
            val sameTruth = labeled[i].truthOrder == labeled[j].truthOrder
            val sameGroup = predicted.getValue(labeled[i].id) == predicted.getValue(labeled[j].id)
            if (sameTruth) positive++ else negative++
            if (sameTruth && sameGroup) tp++
            else if (sameTruth) fn++
            else if (sameGroup) fp++
        }
        if (positive == 0L || negative == 0L) return LifePairMetrics(labeled.size, positive, negative, tp, fp, fn, null, null, null)
        val precision = if (tp + fp == 0L) 0.0 else tp.toDouble() / (tp + fp)
        val recall = tp.toDouble() / positive
        val f1 = if (precision + recall == 0.0) 0.0 else 2 * precision * recall / (precision + recall)
        return LifePairMetrics(labeled.size, positive, negative, tp, fp, fn, precision, recall, f1)
    }
}
