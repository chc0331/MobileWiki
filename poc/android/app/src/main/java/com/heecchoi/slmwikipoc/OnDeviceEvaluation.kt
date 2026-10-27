package com.heecchoi.slmwikipoc

import android.content.Context
import org.json.JSONObject

data class RetrievalMetrics(val recallAt1: Double, val recallAt5: Double, val mrr: Double)
data class PairMetrics(val precision: Double, val recall: Double, val f1: Double, val falseMerges: Int)
data class MixedGroup(val group: NoteGroup, val eventCount: Int)
data class EvaluationResult(
    val retrieval: Map<SearchMode, RetrievalMetrics>,
    val sampledPairs: PairMetrics,
    val allEventPairs: PairMetrics,
    val mixedGroups: List<MixedGroup>,
    val documentCount: Int,
    val queryCount: Int,
    val pairCount: Int,
    val threshold: Double,
    val datasetRevision: String,
)

class OnDeviceEvaluation(private val context: Context, private val store: LocalStore, private val model: EmbeddingModel) {
    fun run(progress: (Int, Int) -> Unit): EvaluationResult {
        val root = JSONObject(context.assets.open("evaluation.json").bufferedReader().use { it.readText() })
        val manifest = root.getJSONObject("manifest")
        val rawDocuments = root.getJSONArray("documents")
        val documents = buildList {
            for (index in 0 until rawDocuments.length()) {
                val row = rawDocuments.getJSONObject(index)
                val id = row.getString("doc_id")
                val cached = store.evalEmbedding(id, model.modelHash)
                val vectors = cached?.let { NoteVectors(it.searchVectors, it.clusterVector) }
                    ?: model.embedNote(row.getString("title"), row.getString("text")).also {
                        store.saveEvalEmbedding(id, model.modelHash, it.searchChunks, it.cluster)
                    }
                add(
                    Note(
                        id = id,
                        workspace = "evaluation",
                        eventId = row.getString("event_id"),
                        title = row.getString("title"),
                        text = row.getString("text"),
                        createdAt = 0,
                        updatedAt = 0,
                        searchVectors = vectors.searchChunks,
                        clusterVector = vectors.cluster,
                    ),
                )
                if ((index + 1) % 10 == 0 || index + 1 == rawDocuments.length()) progress(index + 1, rawDocuments.length())
            }
        }
        val queries = root.getJSONArray("queries")
        val ranks = SearchMode.values().associateWith { mutableListOf<Int?>() }
        for (index in 0 until queries.length()) {
            val query = queries.getJSONObject(index)
            val relevant = buildSet {
                val values = query.getJSONArray("relevant_doc_ids")
                for (i in 0 until values.length()) add(values.getString(i))
            }
            val queryVector = model.embedQuery(query.getString("text"))
            val rankings = NoteSearch.searchModes(documents, query.getString("text"), queryVector, documents.size)
            for (mode in SearchMode.values()) {
                val found = rankings.getValue(mode).take(5).map { it.note.id }
                val firstRank = found.indexOfFirst(relevant::contains).takeIf { it >= 0 }?.plus(1)
                ranks.getValue(mode).add(firstRank)
            }
        }
        val retrieval = ranks.mapValues { (_, values) ->
            RetrievalMetrics(
                recallAt1 = values.count { it == 1 }.toDouble() / values.size,
                recallAt5 = values.count { it != null }.toDouble() / values.size,
                mrr = values.filterNotNull().sumOf { 1.0 / it } / values.size,
            )
        }
        val groups = AverageLinkageGroups.cluster(documents, manifest.getDouble("cluster_threshold"))
        val groupIds = groups.flatMap { group -> group.notes.map { it.id to group.id } }.toMap()
        val rawPairs = root.getJSONArray("pairs")
        var pairTp = 0
        var pairFp = 0
        var pairFn = 0
        for (index in 0 until rawPairs.length()) {
            val pair = rawPairs.getJSONObject(index)
            val related = pair.getBoolean("related")
            val sameGroup = groupIds[pair.getString("left_id")] == groupIds[pair.getString("right_id")]
            when {
                related && sameGroup -> pairTp++
                related -> pairFn++
                sameGroup -> pairFp++
            }
        }
        val sampled = pairMetrics(pairTp, pairFp, pairFn)
        val eventCounts = documents.groupingBy { it.eventId }.eachCount()
        val clusterCounts = groups.associate { it.id to it.notes.size }
        val jointCounts = groups.sumOf { group ->
            group.notes.groupingBy { it.eventId }.eachCount().values.sumOf(::chooseTwo)
        }
        val truePairs = eventCounts.values.sumOf(::chooseTwo)
        val predictedPairs = clusterCounts.values.sumOf(::chooseTwo)
        val allEvents = pairMetrics(jointCounts, predictedPairs - jointCounts, truePairs - jointCounts)
        val mixedGroups = groups.mapNotNull { group ->
            val eventCount = group.notes.mapNotNull { it.eventId }.distinct().size
            if (eventCount > 1) MixedGroup(group, eventCount) else null
        }.sortedWith(compareByDescending<MixedGroup> { it.group.notes.size }.thenByDescending { it.eventCount }).take(8)
        return EvaluationResult(
            retrieval = retrieval,
            sampledPairs = sampled,
            allEventPairs = allEvents,
            mixedGroups = mixedGroups,
            documentCount = documents.size,
            queryCount = queries.length(),
            pairCount = rawPairs.length(),
            threshold = manifest.getDouble("cluster_threshold"),
            datasetRevision = manifest.getString("dataset_revision"),
        )
    }

    private fun pairMetrics(tp: Int, fp: Int, fn: Int): PairMetrics {
        val precision = if (tp + fp == 0) 0.0 else tp.toDouble() / (tp + fp)
        val recall = if (tp + fn == 0) 0.0 else tp.toDouble() / (tp + fn)
        val f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall)
        return PairMetrics(precision, recall, f1, fp)
    }

    private fun chooseTwo(count: Int) = count * (count - 1) / 2
}
