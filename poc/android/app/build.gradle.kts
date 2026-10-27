import groovy.json.JsonSlurper
import java.security.MessageDigest

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.heecchoi.slmwikipoc"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.heecchoi.slmwikipoc"
        minSdk = 31
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-poc"
        ndk { abiFilters += "arm64-v8a" }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }
    androidResources { noCompress += "onnx" }
    packaging { jniLibs.useLegacyPackaging = false }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.08.00"))
    implementation("androidx.activity:activity-compose:1.11.0")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.9.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")
    implementation("com.zhufucdev.hgtk:core-android:0.1.1")

    debugImplementation("androidx.compose.ui:ui-tooling")
    testImplementation("junit:junit:4.13.2")
}

val verifyPocAssets by tasks.registering {
    val modelDir = layout.projectDirectory.dir("src/main/assets/model")
    val evalFile = layout.projectDirectory.file("src/main/assets/evaluation.json")
    val sampleFile = layout.projectDirectory.file("src/main/assets/sample_notes.json")
    val evalManifestFile = layout.projectDirectory.file("src/main/assets/evaluation_manifest.json")
    inputs.dir(modelDir)
    inputs.files(evalFile, sampleFile, evalManifestFile)
    doLast {
        val onnx = modelDir.file("embedding_encoder.onnx").asFile
        val tokenizer = modelDir.file("tokenizer.json").asFile
        val modelManifestFile = modelDir.file("embedding_encoder.manifest.json").asFile
        listOf(onnx, tokenizer, modelManifestFile, evalFile.asFile, sampleFile.asFile, evalManifestFile.asFile).forEach {
            check(it.isFile) { "Missing POC asset: ${it.path}. Run the documented model/data packaging scripts first." }
        }
        fun sha256(file: java.io.File): String {
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
        val json = JsonSlurper()
        val model = json.parse(modelManifestFile) as Map<*, *>
        check(model["base_revision"] == "914f7f89142e33e77833254d9c9b90c3cef7303b") { "Wrong base model revision in Android model manifest" }
        check(model["source_model_repo"] == "heecchoi/slm-wiki-embeddinggemma2-ko-lora-v1") { "Wrong source model repository in Android model manifest" }
        check(model["source_model_revision"] == "f51442c47345d2f3e9113c5ad77057e42d14ea59") { "Wrong source model revision in Android model manifest" }
        check(model["model_hub_repo"] == "heecchoi/slm-wiki-embeddinggemma2-ko-lora-v1") { "Wrong Android model repository in Android model manifest" }
        check(model["model_hub_revision"] == "dbb3bdd252f9034681cb65efee5d7f4b8cbef7bf") { "Wrong Android model artifact revision in Android model manifest" }
        check(model["adapter_checkpoint"] == "checkpoint-250") { "Android model is not the development-selected trained adapter" }
        check(model["runtime"] == "ONNX Runtime Android CPU") { "Expected the verified ONNX Runtime Android CPU model" }
        check(model["precision"] == "int8_rowwise_token_embedding_float32_transformer") { "Expected validated row-wise int8 embedding quantization" }
        check((model["onnx_size_bytes"] as Number).toLong() == onnx.length()) { "ONNX model size does not match its manifest" }
        check((model["tokenizer_size_bytes"] as Number).toLong() == tokenizer.length()) { "Tokenizer size does not match its manifest" }
        check(model["onnx_sha256"] == sha256(onnx)) { "ONNX model hash does not match its manifest" }
        check(model["tokenizer_sha256"] == sha256(tokenizer)) { "Tokenizer hash does not match its manifest" }
        val modelBytes = onnx.length() + tokenizer.length()
        check(modelBytes <= 1_000_000_000L) { "Model and tokenizer exceed the 1 GB POC limit: $modelBytes bytes" }
        val evaluation = json.parse(evalFile.asFile) as Map<*, *>
        val evaluationManifest = evaluation["manifest"] as Map<*, *>
        check((evaluation["documents"] as List<*>).size == 1_000) { "Expected 1,000 fixed test documents" }
        check((evaluation["queries"] as List<*>).size == 140) { "Expected 140 fixed test queries" }
        check((evaluation["pairs"] as List<*>).size == 100) { "Expected 100 fixed test pairs" }
        check(evaluationManifest["selected_checkpoint"] == "checkpoint-250") { "Evaluation data and selected adapter do not match" }
        check(evaluationManifest["dataset_revision"] == "ddba19537f56966c5fbe352707b93e71f57ccdf0") { "Unexpected fixed test dataset revision" }
        check(evaluationManifest["human_review"]?.toString()?.contains("pending") == true) { "Provisional metrics must remain labeled pending human review" }
        val bundledEvalManifest = json.parse(evalManifestFile.asFile) as Map<*, *>
        check(bundledEvalManifest["evaluation_json_sha256"] == sha256(evalFile.asFile)) { "Evaluation data hash mismatch" }
        check(bundledEvalManifest["sample_notes_sha256"] == sha256(sampleFile.asFile)) { "Sample note hash mismatch" }
    }
}

tasks.named("preBuild").configure { dependsOn(verifyPocAssets) }

val verifyOfflineDebugManifest by tasks.registering {
    dependsOn("processDebugManifest")
    val mergedManifest = layout.buildDirectory.file("intermediates/merged_manifest/debug/processDebugMainManifest/AndroidManifest.xml")
    inputs.file(mergedManifest)
    doLast {
        val xml = mergedManifest.get().asFile.readText()
        listOf("android.permission.INTERNET", "android.permission.ACCESS_NETWORK_STATE").forEach { permission ->
            check(!xml.contains("android:name=\"$permission\"")) { "Offline POC APK must not request $permission" }
        }
    }
}

tasks.matching { it.name == "assembleDebug" }.configureEach { dependsOn(verifyOfflineDebugManifest) }
