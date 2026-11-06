package com.heecchoi.slmwikipoc

import java.util.Random
import kotlin.math.sqrt

data class ProjectedDocument(val noteId: String, val x: Float, val y: Float)

data class ExperimentClusteringMetrics(
    val precision: Double?,
    val recall: Double?,
    val f1: Double?,
    val labeledDocumentCount: Int,
    val labeledPairCount: Int,
    val labelCoverage: Double,
    val falseMerges: Int,
)

object ExperimentClusteringMetricsCalculator {
    fun calculate(notes: List<Note>, groups: List<NoteGroup>): ExperimentClusteringMetrics {
        val labeled = notes.filter { !it.groundTruthGroup.isNullOrBlank() }
        val labeledCount = labeled.size
        val pairCount = labeledCount * (labeledCount - 1) / 2
        val coverage = if (notes.isEmpty()) 0.0 else labeledCount.toDouble() / notes.size
        if (pairCount == 0) return ExperimentClusteringMetrics(null, null, null, labeledCount, pairCount, coverage, 0)

        val predictedGroup = groups.flatMap { group -> group.notes.map { it.id to group.id } }.toMap()
        var truePositive = 0
        var falsePositive = 0
        var falseNegative = 0
        for (leftIndex in 0 until labeled.lastIndex) {
            for (rightIndex in leftIndex + 1 until labeled.size) {
                val left = labeled[leftIndex]
                val right = labeled[rightIndex]
                val sameTruth = left.groundTruthGroup!!.trim() == right.groundTruthGroup!!.trim()
                val samePrediction = predictedGroup[left.id] == predictedGroup[right.id]
                when {
                    sameTruth && samePrediction -> truePositive++
                    sameTruth -> falseNegative++
                    samePrediction -> falsePositive++
                }
            }
        }
        val precision = if (truePositive + falsePositive == 0) 0.0 else truePositive.toDouble() / (truePositive + falsePositive)
        val recall = if (truePositive + falseNegative == 0) 0.0 else truePositive.toDouble() / (truePositive + falseNegative)
        val f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall)
        return ExperimentClusteringMetrics(precision, recall, f1, labeledCount, pairCount, coverage, falsePositive)
    }
}

object PcaProjector {
    private const val ITERATIONS = 36

    fun project(notes: List<Note>, checkCancelled: () -> Unit = {}): List<ProjectedDocument> {
        if (notes.isEmpty()) return emptyList()
        if (notes.size == 1) return listOf(ProjectedDocument(notes.single().id, 0.5f, 0.5f))
        val dimensions = notes.first().clusterVector.size
        val mean = DoubleArray(dimensions) { dimension -> notes.sumOf { it.clusterVector[dimension].toDouble() } / notes.size }
        val centered = Array(notes.size) { row ->
            DoubleArray(dimensions) { dimension -> notes[row].clusterVector[dimension].toDouble() - mean[dimension] }
        }
        val first = principalComponent(centered, emptyList(), 41L, checkCancelled)
        val second = principalComponent(centered, listOfNotNull(first), 83L, checkCancelled)
        val x = centered.map { row -> first?.let { dot(row, it) } ?: 0.0 }
        val y = centered.map { row -> second?.let { dot(row, it) } ?: 0.0 }
        val xRange = x.maxOrNull()!! - x.minOrNull()!!
        val yRange = y.maxOrNull()!! - y.minOrNull()!!
        return notes.indices.map { index ->
            val normalizedX = if (xRange < 1e-12) 0.5 else (x[index] - x.minOrNull()!!) / xRange
            val normalizedY = if (yRange < 1e-12) 0.5 else (y[index] - y.minOrNull()!!) / yRange
            ProjectedDocument(notes[index].id, normalizedX.toFloat(), normalizedY.toFloat())
        }
    }

    private fun principalComponent(rows: Array<DoubleArray>, componentsToRemove: List<DoubleArray>, seed: Long, checkCancelled: () -> Unit): DoubleArray? {
        val dimensions = rows.firstOrNull()?.size ?: return null
        val random = Random(seed)
        var vector = DoubleArray(dimensions) { random.nextDouble() - 0.5 }
        orthogonalize(vector, componentsToRemove)
        if (!normalize(vector)) return null
        repeat(ITERATIONS) {
            checkCancelled()
            val projections = DoubleArray(rows.size) { row -> dot(rows[row], vector) }
            val next = DoubleArray(dimensions)
            rows.indices.forEach { rowIndex ->
                val scale = projections[rowIndex]
                rows[rowIndex].indices.forEach { dimension -> next[dimension] += rows[rowIndex][dimension] * scale }
            }
            orthogonalize(next, componentsToRemove)
            if (!normalize(next)) return null
            vector = next
        }
        val largest = vector.indices.maxByOrNull { kotlin.math.abs(vector[it]) } ?: return null
        if (vector[largest] < 0) vector.indices.forEach { vector[it] = -vector[it] }
        return vector
    }

    private fun orthogonalize(vector: DoubleArray, components: List<DoubleArray>) {
        components.forEach { component ->
            val projection = dot(vector, component)
            vector.indices.forEach { index -> vector[index] -= projection * component[index] }
        }
    }

    private fun normalize(vector: DoubleArray): Boolean {
        val norm = sqrt(vector.sumOf { it * it })
        if (norm < 1e-12) return false
        vector.indices.forEach { vector[it] /= norm }
        return true
    }

    private fun dot(left: DoubleArray, right: DoubleArray): Double = left.indices.sumOf { left[it] * right[it] }
}
