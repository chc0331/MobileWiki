package com.heecchoi.slmwikipoc

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer

class OnDeviceSpeech(
    private val context: Context,
    private val onStatus: (String) -> Unit,
    private val onPartial: (String) -> Unit,
    private val onResult: (String) -> Unit,
    private val onError: (String) -> Unit,
) {
    private var recognizer: SpeechRecognizer? = null
    private var generation = 0

    fun start() {
        cancel()
        if (!SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) {
            onError("이 기기에서는 기기 내 음성 인식을 사용할 수 없습니다. 직접 입력해 주세요.")
            return
        }
        val token = ++generation
        try {
            val instance = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
            recognizer = instance
            instance.setRecognitionListener(object : RecognitionListener {
                private fun current() = token == generation
                override fun onReadyForSpeech(params: Bundle?) { if (current()) onStatus("말씀해 주세요.") }
                override fun onBeginningOfSpeech() { if (current()) onStatus("듣는 중입니다.") }
                override fun onRmsChanged(rmsdB: Float) = Unit
                override fun onBufferReceived(buffer: ByteArray?) = Unit
                override fun onEndOfSpeech() { if (current()) onStatus("음성을 글로 바꾸는 중입니다.") }
                override fun onError(error: Int) {
                    if (!current()) return
                    release()
                    onError(when (error) {
                        SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "마이크 권한이 필요합니다. 설정에서 허용해 주세요."
                        SpeechRecognizer.ERROR_NO_MATCH, SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "음성을 인식하지 못했습니다. 다시 말씀하거나 직접 입력해 주세요."
                        SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED, SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> "기기 내 한국어 음성 인식이 준비되지 않았습니다. 직접 입력해 주세요."
                        SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "다른 음성 인식이 사용 중입니다. 잠시 뒤 다시 시도해 주세요."
                        else -> "기기 내 음성 인식을 완료하지 못했습니다. 다시 시도하거나 직접 입력해 주세요."
                    })
                }
                override fun onResults(results: Bundle?) {
                    if (!current()) return
                    val text = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty().trim()
                    release()
                    if (text.isEmpty()) onError("인식된 글이 없습니다. 다시 말씀하거나 직접 입력해 주세요.") else onResult(text)
                }
                override fun onPartialResults(partialResults: Bundle?) {
                    if (current()) partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()?.let(onPartial)
                }
                override fun onEvent(eventType: Int, params: Bundle?) = Unit
            })
            onStatus("음성 인식을 준비하는 중입니다.")
            instance.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                putExtra(RecognizerIntent.EXTRA_LANGUAGE, "ko-KR")
                putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
            })
        } catch (_: Exception) {
            release()
            onError("기기 내 음성 인식을 시작하지 못했습니다. 직접 입력해 주세요.")
        }
    }

    fun stop() {
        try { recognizer?.stopListening() } catch (_: Exception) {
            release()
            onError("음성 인식을 마치지 못했습니다. 다시 시도해 주세요.")
        }
    }

    fun cancel() {
        generation++
        runCatching { recognizer?.cancel() }
        release()
    }

    private fun release() {
        generation++
        runCatching { recognizer?.destroy() }
        recognizer = null
    }
}
