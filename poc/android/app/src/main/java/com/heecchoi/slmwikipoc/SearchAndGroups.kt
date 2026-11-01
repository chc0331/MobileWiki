package com.heecchoi.slmwikipoc

import java.util.PriorityQueue
import java.util.Locale
import kotlin.math.ln
import kotlin.math.sqrt

enum class SearchMode(val label: String) { DENSE("임베딩"), LEXICAL("낱말"), HYBRID("결합") }

data class SearchHit(val note: Note, val score: Double, val denseScore: Double, val lexicalScore: Double)
data class NoteGroup(val id: Int, val label: String, val notes: List<Note>)

object NoteSearch {
    fun search(notes: List<Note>, query: String, queryVector: FloatArray, mode: SearchMode, limit: Int = 40): List<SearchHit> {
        return searchModes(notes, query, queryVector, limit).getValue(mode)
    }

    fun searchModes(notes: List<Note>, query: String, queryVector: FloatArray, limit: Int = 40): Map<SearchMode, List<SearchHit>> {
        if (notes.isEmpty()) return SearchMode.values().associateWith { emptyList() }
        val dense = notes.associate { note -> note.id to note.searchVectors.maxOf { cosine(queryVector, it) } }
        val lexical = lexicalScores(notes, query)
        val denseRank = rank(notes, dense)
        val lexicalRank = rank(notes, lexical)
        val densePosition = denseRank.withIndex().associate { it.value.id to it.index + 1 }
        val lexicalPosition = lexicalRank.withIndex().associate { it.value.id to it.index + 1 }
        return SearchMode.values().associateWith { mode -> notes.map { note ->
            val d = dense.getValue(note.id)
            val l = lexical.getValue(note.id)
            val score = when (mode) {
                SearchMode.DENSE -> d
                SearchMode.LEXICAL -> l
                SearchMode.HYBRID -> 1.0 / (60 + densePosition.getValue(note.id)) + 1.0 / (60 + lexicalPosition.getValue(note.id))
            }
            SearchHit(note, score, d, l)
        }.sortedWith(compareByDescending<SearchHit> { it.score }.thenBy { it.note.id }).take(limit) }
    }

    private fun lexicalScores(notes: List<Note>, query: String): Map<String, Double> {
        val docs = notes.map { "${it.title} ${it.text}".lowercase(Locale.ROOT) }
        val terms = docs.map(::ngrams)
        val df = HashMap<String, Int>()
        terms.forEach { frequencies -> frequencies.keys.forEach { term -> df[term] = (df[term] ?: 0) + 1 } }
        val queryTerms = ngrams(query.lowercase(Locale.ROOT))
        val queryWeighted = queryTerms.mapValues { (term, tf) -> tf * (ln((1.0 + notes.size) / (1.0 + (df[term] ?: 0))) + 1.0) }
        val queryNorm = sqrt(queryWeighted.values.sumOf { it * it })
        return notes.indices.associate { index ->
            val weights = terms[index].mapValues { (term, tf) -> tf * (ln((1.0 + notes.size) / (1.0 + (df[term] ?: 0))) + 1.0) }
            val norm = sqrt(weights.values.sumOf { it * it })
            val dot = queryWeighted.entries.sumOf { (term, value) -> value * (weights[term] ?: 0.0) }
            notes[index].id to if (norm == 0.0 || queryNorm == 0.0) 0.0 else dot / (norm * queryNorm)
        }
    }

    private fun ngrams(text: String): Map<String, Int> {
        val chars = text.codePoints().toArray()
        val frequencies = HashMap<String, Int>()
        for (size in 2..4) {
            for (start in 0..(chars.size - size)) {
                val gram = String(chars, start, size)
                frequencies[gram] = (frequencies[gram] ?: 0) + 1
            }
        }
        return frequencies
    }

    private fun rank(notes: List<Note>, scores: Map<String, Double>) = notes.sortedWith(
        compareByDescending<Note> { scores[it.id] ?: 0.0 }.thenBy { it.id },
    )

    fun cosine(left: FloatArray, right: FloatArray): Double {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        for (i in left.indices) {
            dot += left[i].toDouble() * right[i]
            leftNorm += left[i].toDouble() * left[i]
            rightNorm += right[i].toDouble() * right[i]
        }
        return if (leftNorm == 0.0 || rightNorm == 0.0) 0.0 else dot / sqrt(leftNorm * rightNorm)
    }
}

object AverageLinkageGroups {
    private data class Candidate(val distance: Double, val left: Int, val right: Int)
    private data class MergeNode(val leaf: Int = -1, val left: Int = -1, val right: Int = -1, val distance: Double = Double.NaN)

    fun clusterLarge(notes: List<Note>, threshold: Double, checkCancelled: () -> Unit = {}): List<NoteGroup> {
        val count = notes.size
        if (count <= 1) return cluster(notes, threshold)
        val pairCount = count.toLong() * (count - 1) / 2
        val requiredBytes = pairCount * Double.SIZE_BYTES
        check(pairCount <= Int.MAX_VALUE && requiredBytes < Runtime.getRuntime().maxMemory() / 2) {
            "문서 ${count}개의 정확한 평균 연결 군집에 필요한 메모리가 부족합니다."
        }
        val distances = DoubleArray(pairCount.toInt())
        fun position(a: Int, b: Int): Int {
            val low = minOf(a, b)
            val high = maxOf(a, b)
            return (low.toLong() * (2L * count - low - 1) / 2 + high - low - 1).toInt()
        }
        fun distance(a: Int, b: Int) = distances[position(a, b)]
        fun setDistance(a: Int, b: Int, value: Double) { distances[position(a, b)] = value }
        for (a in 0 until count) {
            checkCancelled()
            for (b in a + 1 until count) {
                setDistance(a, b, (1.0 - NoteSearch.cosine(notes[a].clusterVector, notes[b].clusterVector)).coerceIn(0.0, 2.0))
            }
        }
        val active = BooleanArray(count) { true }
        val sizes = IntArray(count) { 1 }
        val nodes = MutableList(count) { index -> MergeNode(leaf = index) }
        val nodeAt = IntArray(count) { it }
        val chain = ArrayList<Int>()
        var remaining = count
        while (remaining > 1) {
            checkCancelled()
            if (chain.isEmpty()) chain += active.indexOfFirst { it }
            val a = chain.last()
            var closest = -1
            var closestDistance = Double.POSITIVE_INFINITY
            for (b in 0 until count) if (active[b] && b != a) {
                val value = distance(a, b)
                if (value < closestDistance || (value == closestDistance && b < closest)) {
                    closest = b
                    closestDistance = value
                }
            }
            if (chain.size >= 2 && closest == chain[chain.lastIndex - 1]) {
                val b = closest
                val keep = minOf(a, b)
                val drop = maxOf(a, b)
                val combinedSize = sizes[a] + sizes[b]
                nodes += MergeNode(left = nodeAt[a], right = nodeAt[b], distance = closestDistance)
                nodeAt[keep] = nodes.lastIndex
                for (other in 0 until count) if (active[other] && other != a && other != b) {
                    setDistance(keep, other, (sizes[a] * distance(a, other) + sizes[b] * distance(b, other)) / combinedSize)
                }
                sizes[keep] = combinedSize
                active[drop] = false
                remaining--
                chain.clear()
                chain += keep
            } else if (closest in chain) {
                chain.clear()
                chain += closest
            } else {
                chain += closest
            }
        }
        val cut = ArrayList<Int>()
        val pending = ArrayDeque<Int>()
        pending.add(nodeAt[active.indexOfFirst { it }])
        while (pending.isNotEmpty()) {
            val index = pending.removeLast()
            val node = nodes[index]
            if (node.leaf >= 0 || node.distance <= threshold) cut += index
            else { pending.add(node.left); pending.add(node.right) }
        }
        return cut.map { index ->
            val leafIds = ArrayList<Int>()
            pending.add(index)
            while (pending.isNotEmpty()) {
                val node = nodes[pending.removeLast()]
                if (node.leaf >= 0) leafIds += node.leaf
                else { pending.add(node.left); pending.add(node.right) }
            }
            val groupNotes = leafIds.sorted().map(notes::get)
            NoteGroup(index, groupNotes.first().title.ifBlank { "이름 미정" }, groupNotes)
        }.sortedWith(compareByDescending<NoteGroup> { it.notes.size }.thenBy { it.label })
    }

    fun cluster(notes: List<Note>, threshold: Double): List<NoteGroup> {
        if (notes.isEmpty()) return emptyList()
        if (notes.size == 1) return listOf(NoteGroup(0, notes[0].title.ifBlank { "이름 미정" }, notes))
        val count = notes.size
        val capacity = count * 2
        val distance = Array(capacity) { DoubleArray(capacity) }
        val sizes = IntArray(capacity) { 1 }
        val members = Array(capacity) { mutableListOf<Int>() }
        val active = BooleanArray(capacity)
        repeat(count) { index -> members[index].add(index); active[index] = true }
        val queue = PriorityQueue<Candidate>(compareBy<Candidate> { it.distance }.thenBy { it.left }.thenBy { it.right })
        for (left in 0 until count) for (right in left + 1 until count) {
            val value = (1.0 - NoteSearch.cosine(notes[left].clusterVector, notes[right].clusterVector)).coerceIn(0.0, 2.0)
            distance[left][right] = value
            distance[right][left] = value
            queue += Candidate(value, left, right)
        }
        var next = count
        while (queue.isNotEmpty()) {
            val candidate = queue.remove()
            val left = candidate.left
            val right = candidate.right
            if (!active[left] || !active[right]) continue
            if (candidate.distance > threshold) break
            active[left] = false
            active[right] = false
            active[next] = true
            sizes[next] = sizes[left] + sizes[right]
            members[next].addAll(members[left])
            members[next].addAll(members[right])
            for (other in 0 until next) if (active[other]) {
                val nextDistance = (sizes[left] * distance[left][other] + sizes[right] * distance[right][other]) / sizes[next]
                distance[next][other] = nextDistance
                distance[other][next] = nextDistance
                val a = minOf(next, other)
                val b = maxOf(next, other)
                queue += Candidate(nextDistance, a, b)
            }
            next++
        }
        return (0 until next).filter { active[it] }.map { id ->
            val groupNotes = members[id].map(notes::get)
            NoteGroup(id, groupNotes.first().title.ifBlank { "이름 미정" }, groupNotes)
        }.sortedWith(compareByDescending<NoteGroup> { it.notes.size }.thenBy { it.label })
    }
}
