package com.heecchoi.slmwikipoc

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.encryption.InvalidPasswordException
import com.tom_roush.pdfbox.text.PDFTextStripper
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.util.UUID

data class ParsedDocument(
    val id: String,
    val fileName: String,
    val mimeType: String,
    val title: String,
    val text: String,
    val pageTexts: List<String>,
    val emptyPageNumbers: List<Int>,
    val temporaryPdfPath: String?,
)

data class ImportPreviewItem(
    val document: ParsedDocument?,
    val fileName: String,
    val error: String? = null,
    val groundTruthGroup: String = "",
)

object DocumentImportParser {
    const val MAX_FILE_BYTES = 10L * 1024 * 1024
    private const val MAX_TEXT_CHARS = 20_000
    private const val MAX_PDF_PAGES = 50
    private const val MAX_PDF_TEXT_CHARS = 100_000

    @Volatile private var pdfBoxInitialized = false

    fun parse(context: Context, uri: Uri): ParsedDocument = parse(context, uri, displayName(context, uri))

    fun parseFile(context: Context, file: File): ParsedDocument = parse(context, Uri.fromFile(file), file.name)

    private fun parse(context: Context, uri: Uri, fileName: String): ParsedDocument {
        val extension = fileName.substringAfterLast('.', "").lowercase()
        val id = UUID.randomUUID().toString()
        return when (extension) {
            "txt", "md" -> parseText(context, uri, id, fileName, extension)
            "pdf" -> parsePdf(context, uri, id, fileName)
            else -> error("지원하지 않는 파일입니다. TXT, Markdown, 텍스트 PDF만 가져올 수 있습니다.")
        }
    }

    fun persistSource(document: ParsedDocument, directory: File): DocumentSource {
        val storedPath = document.temporaryPdfPath?.let { temporaryPath ->
            val temporary = File(temporaryPath)
            val target = File(directory.apply { mkdirs() }, "${document.id}.pdf")
            temporary.copyTo(target, overwrite = true)
            target.absolutePath
        }
        return DocumentSource(document.fileName, document.mimeType, storedPath, document.pageTexts)
    }

    fun cleanup(document: ParsedDocument) {
        document.temporaryPdfPath?.let { File(it).delete() }
    }

    private fun parseText(context: Context, uri: Uri, id: String, fileName: String, extension: String): ParsedDocument {
        val bytes = context.contentResolver.openInputStream(uri)?.use { input ->
            val output = ByteArrayOutputStream()
            val buffer = ByteArray(8192)
            var total = 0L
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                total += read
                require(total <= MAX_FILE_BYTES) { "파일 크기가 10MB를 넘습니다." }
                output.write(buffer, 0, read)
            }
            output.toByteArray()
        } ?: error("파일을 열 수 없습니다.")
        val text = try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(java.nio.ByteBuffer.wrap(bytes)).toString().trim()
        } catch (_: java.nio.charset.CharacterCodingException) {
            error("UTF-8 문서가 아닙니다.")
        }
        require(text.isNotEmpty()) { "문서에 글이 없습니다." }
        require(text.length <= MAX_TEXT_CHARS) { "TXT·Markdown은 20,000자까지 가져올 수 있습니다." }
        return ParsedDocument(
            id = id,
            fileName = fileName,
            mimeType = context.contentResolver.getType(uri) ?: if (extension == "md") "text/markdown" else "text/plain",
            title = fileName.substringBeforeLast('.', fileName).ifBlank { fileName },
            text = text,
            pageTexts = emptyList(),
            emptyPageNumbers = emptyList(),
            temporaryPdfPath = null,
        )
    }

    private fun parsePdf(context: Context, uri: Uri, id: String, fileName: String): ParsedDocument {
        val temporary = File.createTempFile("document-import-", ".pdf", context.cacheDir)
        try {
            context.contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(temporary).use { output ->
                    val buffer = ByteArray(8192)
                    var total = 0L
                    while (true) {
                        val read = input.read(buffer)
                        if (read < 0) break
                        total += read
                        require(total <= MAX_FILE_BYTES) { "파일 크기가 10MB를 넘습니다." }
                        output.write(buffer, 0, read)
                    }
                }
            } ?: error("PDF 파일을 열 수 없습니다.")
            if (!pdfBoxInitialized) synchronized(this) {
                if (!pdfBoxInitialized) {
                    PDFBoxResourceLoader.init(context.applicationContext)
                    pdfBoxInitialized = true
                }
            }
            val pageTexts = PDDocument.load(temporary).use { document ->
                require(!document.isEncrypted) { "암호가 설정된 PDF는 가져올 수 없습니다." }
                require(document.numberOfPages in 1..MAX_PDF_PAGES) { "PDF는 1~50쪽까지 가져올 수 있습니다." }
                val stripper = PDFTextStripper()
                (0 until document.numberOfPages).map { pageIndex ->
                    stripper.setStartPage(pageIndex + 1)
                    stripper.setEndPage(pageIndex + 1)
                    stripper.getText(document).trim()
                }
            }
            val totalChars = pageTexts.sumOf(String::length)
            require(totalChars <= MAX_PDF_TEXT_CHARS) { "PDF에서 추출한 글이 100,000자를 넘습니다." }
            require(pageTexts.any(String::isNotBlank)) { "PDF에서 글을 추출하지 못했습니다. 스캔 PDF는 지원하지 않습니다." }
            val emptyPages = pageTexts.mapIndexedNotNull { index, text -> (index + 1).takeIf { text.isBlank() } }
            val body = pageTexts.mapIndexed { index, text -> "[${index + 1}쪽]\n$text" }.joinToString("\n\n")
            return ParsedDocument(
                id = id,
                fileName = fileName,
                mimeType = "application/pdf",
                title = fileName.substringBeforeLast('.', fileName).ifBlank { fileName },
                text = body,
                pageTexts = pageTexts,
                emptyPageNumbers = emptyPages,
                temporaryPdfPath = temporary.absolutePath,
            )
        } catch (error: InvalidPasswordException) {
            temporary.delete()
            throw IllegalArgumentException("암호가 설정된 PDF는 가져올 수 없습니다.", error)
        } catch (error: IOException) {
            temporary.delete()
            throw IllegalArgumentException("PDF 파일이 손상되었거나 읽을 수 없습니다.", error)
        } catch (error: Throwable) {
            temporary.delete()
            throw error
        }
    }

    private fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (index >= 0) cursor.getString(index)?.takeIf(String::isNotBlank)?.let { return it }
            }
        }
        return uri.lastPathSegment?.substringAfterLast('/')?.takeIf(String::isNotBlank) ?: "가져온 문서"
    }
}
