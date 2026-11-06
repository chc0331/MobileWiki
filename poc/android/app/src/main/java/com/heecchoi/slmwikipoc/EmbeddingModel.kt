package com.heecchoi.slmwikipoc

import android.content.Context
import org.json.JSONObject
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import tokenizers.Tokenizer
import java.io.File
import java.nio.LongBuffer
import java.security.MessageDigest
import kotlin.math.sqrt

data class NoteVectors(val searchChunks: List<FloatArray>, val cluster: FloatArray)

object BodyChunking {
    fun ranges(bodyTokens: Int, prefixTokens: Int): List<IntRange> {
        val capacity = minOf(420, 509 - prefixTokens)
        require(capacity > 0 && capacity >= minOf(64, bodyTokens)) { "제목이 너무 길어 본문을 처리할 수 없습니다. 제목을 줄여 주세요." }
        if (bodyTokens == 0) return listOf(0 until 0)
        val overlap = minOf(64, capacity / 2)
        val stride = capacity - overlap
        val chunks = mutableListOf<IntRange>()
        var start = 0
        while (start < bodyTokens) {
            val end = minOf(start + capacity, bodyTokens)
            chunks += start until end
            if (end == bodyTokens) break
            start += stride
        }
        return chunks
    }
}

class EmbeddingModel(private val context: Context) : AutoCloseable {
    private var session: OrtSession? = null
    private val environment = OrtEnvironment.getEnvironment()
    private var tokenizer: Tokenizer? = null
    lateinit var modelHash: String
        private set

    @Synchronized
    fun load() {
        if (session != null) return
        val manifest = JSONObject(context.assets.open("model/embedding_encoder.manifest.json").bufferedReader().use { it.readText() })
        val directory = File(context.noBackupFilesDir, "embedding-model").apply { mkdirs() }
        val modelFile = materialize("model/embedding_encoder.onnx", File(directory, "embedding_encoder.onnx"))
        val tokenizerFile = materialize("model/tokenizer.json", File(directory, "tokenizer.json"))
        modelHash = sha256(modelFile)
        check(modelHash == manifest.getString("onnx_sha256")) { "ONNX 모델 해시가 manifest와 다릅니다." }
        check(sha256(tokenizerFile) == manifest.getString("tokenizer_sha256")) { "토크나이저 해시가 manifest와 다릅니다." }
        check(manifest.getString("adapter_checkpoint") == "checkpoint-250") { "예상한 학습 어댑터가 아닙니다." }
        check(manifest.getString("runtime") == "ONNX Runtime Android CPU") { "검증된 ONNX Runtime 모델이 아닙니다." }
        check(manifest.getString("precision") == "int8_rowwise_token_embedding_float32_transformer") { "검증된 행 단위 int8 모델이 아닙니다." }
        check(modelFile.length() + tokenizerFile.length() <= 1_000_000_000L) { "모델 파일 합계가 1GB 제한을 넘습니다." }
        val options = OrtSession.SessionOptions()
        val loaded = try {
            options.setIntraOpNumThreads(4)
            environment.createSession(modelFile.absolutePath, options)
        } finally {
            options.close()
        }
        try {
            check(loaded.inputNames == setOf("input_ids", "attention_mask") && loaded.outputNames.contains("embeddings")) { "예상한 임베딩 모델 입출력이 아닙니다." }
            val loadedTokenizer = Tokenizer.fromFile(tokenizerFile.absolutePath)
            tokenizer = loadedTokenizer
            session = loaded
        } catch (error: Throwable) {
            loaded.close()
            throw error
        }
    }

    @Synchronized
    fun embedQuery(query: String): FloatArray {
        val text = "task: search result | query: $query"
        return run(tokenize(text, special = true).take(510).toIntArray())
    }

    @Synchronized
    fun embedNote(title: String, text: String): NoteVectors {
        val search = documentChunks(title, text, clustering = false).map(::run)
        val clustering = documentChunks(title, text, clustering = true).map(::run)
        return NoteVectors(search, normalize(mean(clustering)))
    }

    @Synchronized
    fun embedCluster(title: String, text: String, checkCancelled: () -> Unit = {}): FloatArray {
        val vectors = documentChunks(title, text, clustering = true).map { tokens ->
            checkCancelled()
            run(tokens)
        }
        return normalize(mean(vectors))
    }

    private fun documentChunks(title: String, body: String, clustering: Boolean): List<IntArray> {
        val tok = tokenizer ?: error("Model is not loaded")
        val bodyIds = tok.encode(body, withSpecialTokens = false).ids.map(UInt::toInt)
        if (bodyIds.size <= 420) {
            val input = if (clustering) "task: clustering | query: $title\n$body" else "title: $title | text: $body"
            val tokens = tokenize(input, special = true)
            if (tokens.size <= 510) return listOf(tokens.toIntArray())
        }
        val prefix = if (clustering) "task: clustering | query: $title\n" else "title: $title | text: "
        val prefixIds = tok.encode(prefix, withSpecialTokens = false).ids.map(UInt::toInt)
        return BodyChunking.ranges(bodyIds.size, prefixIds.size).map { range ->
            (listOf(2) + prefixIds + bodyIds.subList(range.first, range.last + 1) + 1).toIntArray()
        }
    }

    private fun tokenize(text: String, special: Boolean): List<Int> =
        (tokenizer ?: error("Model is not loaded")).encode(text, withSpecialTokens = special).ids.map(UInt::toInt)

    @Synchronized
    private fun run(tokens: IntArray): FloatArray {
        val engine = session ?: error("Model is not loaded")
        val tokenCount = tokens.size.coerceIn(1, 512)
        val ids = LongArray(tokenCount)
        val mask = LongArray(tokenCount) { 1L }
        tokens.take(512).forEachIndexed { index, token ->
            if (index < tokenCount) ids[index] = token.toLong()
        }
        val shape = longArrayOf(1, tokenCount.toLong())
        OnnxTensor.createTensor(environment, LongBuffer.wrap(ids), shape).use { idsTensor ->
            OnnxTensor.createTensor(environment, LongBuffer.wrap(mask), shape).use { maskTensor ->
                engine.run(mapOf("input_ids" to idsTensor, "attention_mask" to maskTensor), setOf("embeddings")).use { result ->
                    val output = result.get("embeddings").orElseThrow().value as Array<FloatArray>
                    check(output.size == 1 && output[0].size == 768) { "임베딩 출력 차원이 예상과 다릅니다." }
                    return output[0]
                }
            }
        }
    }

    override fun close() {
        session?.close()
        session = null
        tokenizer = null
    }

    private fun materialize(asset: String, destination: File): File {
        if (!destination.exists() || destination.length() == 0L) {
            val temporary = File(destination.parentFile, "${destination.name}.tmp")
            context.assets.open(asset).use { input -> temporary.outputStream().use { output -> input.copyTo(output) } }
            check(temporary.renameTo(destination)) { "모델 파일을 앱 전용 저장소에 복사할 수 없습니다." }
        }
        return destination
    }

    companion object {
        fun from(context: Context) = EmbeddingModel(context.applicationContext)

        private fun sha256(file: File): String {
            val digest = MessageDigest.getInstance("SHA-256")
            file.inputStream().use { input ->
                val buffer = ByteArray(1024 * 1024)
                while (true) {
                    val count = input.read(buffer)
                    if (count < 0) break
                    digest.update(buffer, 0, count)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it) }
        }

        private fun mean(vectors: List<FloatArray>): FloatArray {
            require(vectors.isNotEmpty())
            return FloatArray(768) { index -> vectors.sumOf { it[index].toDouble() }.div(vectors.size).toFloat() }
        }

        private fun normalize(vector: FloatArray): FloatArray {
            val norm = sqrt(vector.sumOf { it.toDouble() * it })
            return if (norm == 0.0) vector else FloatArray(vector.size) { (vector[it] / norm).toFloat() }
        }
    }
}
