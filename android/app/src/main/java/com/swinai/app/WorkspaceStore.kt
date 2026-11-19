package com.swinai.app

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.security.KeyStore
import java.security.MessageDigest
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Account-isolated local project storage.
 *
 * Project JSON is encrypted with an AES-GCM key kept in Android Keystore.
 * No project source files are written to shared/external storage.
 */
internal class WorkspaceStore(context: Context, email: String) {
    private val preferences = context.getSharedPreferences("swin_workspace_v1", Context.MODE_PRIVATE)
    private val keyAlias = "com.swinai.app.workspace.aes.v1"
    private val storageKey = "projects_" + MessageDigest.getInstance("SHA-256")
        .digest(email.trim().lowercase().toByteArray(Charsets.UTF_8))
        .take(16)
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

    fun projects(): List<JSONObject> {
        val source = readRoot().optJSONArray("projects") ?: JSONArray()
        return (0 until source.length()).mapNotNull { index ->
            source.optJSONObject(index)?.let { JSONObject(it.toString()) }
        }.sortedByDescending { it.optLong("updated_at", 0L) }
    }

    fun project(projectId: String): JSONObject? {
        return projects().firstOrNull { it.optString("id") == projectId }
    }

    fun createProject(rawName: String): JSONObject {
        val name = normalizeName(rawName, "project")
        val root = readRoot()
        val projects = root.optJSONArray("projects") ?: JSONArray()
        require(projects.length() < MAX_PROJECTS) { "Достигнут лимит проектов ($MAX_PROJECTS)." }
        require((0 until projects.length()).none {
            projects.optJSONObject(it)?.optString("name")?.equals(name, ignoreCase = true) == true
        }) { "Проект с таким названием уже существует." }

        val now = System.currentTimeMillis() / 1000L
        val item = JSONObject()
            .put("id", "project-" + UUID.randomUUID().toString())
            .put("name", name)
            .put("created_at", now)
            .put("updated_at", now)
            .put("files", JSONArray())
        projects.put(item)
        root.put("projects", projects)
        writeRoot(root)
        return JSONObject(item.toString())
    }

    fun exportProjectData(projectId: String): JSONObject {
        val item = project(projectId) ?: throw IllegalArgumentException("Проект не найден.")
        val exportFiles = JSONArray()
        val files = item.optJSONArray("files") ?: JSONArray()
        for (index in 0 until files.length()) {
            val file = files.optJSONObject(index) ?: continue
            exportFiles.put(
                JSONObject()
                    .put("name", file.optString("name"))
                    .put("language", file.optString("language", "text"))
                    .put("content", file.optString("content", ""))
            )
        }
        return JSONObject()
            .put("format", "swin-ai-project")
            .put("version", 1)
            .put("project_name", item.optString("name"))
            .put("files", exportFiles)
    }

    fun importProject(projectName: String, sourceFiles: List<JSONObject>): JSONObject {
        require(sourceFiles.size <= MAX_FILES_PER_PROJECT) {
            "В архиве максимум $MAX_FILES_PER_PROJECT файлов."
        }
        val created = createProject(projectName)
        val projectId = created.optString("id")
        try {
            for (file in sourceFiles) {
                createFile(
                    projectId,
                    file.optString("name"),
                    file.optString("language", "text"),
                    file.optString("content", "")
                )
            }
            return project(projectId) ?: throw IllegalStateException("Не удалось проверить импорт проекта.")
        } catch (error: Exception) {
            deleteProject(projectId)
            throw error
        }
    }

    fun deleteProject(projectId: String): Boolean {
        val root = readRoot()
        val projects = root.optJSONArray("projects") ?: JSONArray()
        val kept = JSONArray()
        var removed = false
        for (index in 0 until projects.length()) {
            val item = projects.optJSONObject(index) ?: continue
            if (item.optString("id") == projectId) {
                removed = true
            } else {
                kept.put(item)
            }
        }
        if (removed) {
            root.put("projects", kept)
            writeRoot(root)
        }
        return removed
    }

    fun createFile(
        projectId: String,
        rawName: String,
        language: String,
        initialContent: String = ""
    ): JSONObject {
        val name = normalizeName(rawName, "file")
        require(initialContent.length <= MAX_FILE_CHARS) { "Файл слишком большой." }
        val root = readRoot()
        val project = findProject(root, projectId) ?: throw IllegalArgumentException("Проект не найден.")
        val files = project.optJSONArray("files") ?: JSONArray()
        require(files.length() < MAX_FILES_PER_PROJECT) {
            "В проекте максимум $MAX_FILES_PER_PROJECT файлов."
        }
        require((0 until files.length()).none {
            files.optJSONObject(it)?.optString("name")?.equals(name, ignoreCase = true) == true
        }) { "Файл с таким названием уже существует." }

        val now = System.currentTimeMillis() / 1000L
        val item = JSONObject()
            .put("id", "file-" + UUID.randomUUID().toString())
            .put("name", name)
            .put("language", language)
            .put("content", initialContent)
            .put("created_at", now)
            .put("updated_at", now)
        files.put(item)
        project.put("files", files).put("updated_at", now)
        writeRoot(root)
        return JSONObject(item.toString())
    }

    fun saveFile(
        projectId: String,
        fileId: String,
        rawName: String,
        language: String,
        content: String
    ) {
        val name = normalizeName(rawName, "file")
        require(content.length <= MAX_FILE_CHARS) {
            "Файл превышает лимит $MAX_FILE_CHARS символов."
        }
        val root = readRoot()
        val project = findProject(root, projectId) ?: throw IllegalArgumentException("Проект не найден.")
        val files = project.optJSONArray("files") ?: JSONArray()
        require((0 until files.length()).none { index ->
            val file = files.optJSONObject(index)
            file != null &&
                file.optString("id") != fileId &&
                file.optString("name").equals(name, ignoreCase = true)
        }) { "Файл с таким названием уже существует." }

        var found = false
        val now = System.currentTimeMillis() / 1000L
        for (index in 0 until files.length()) {
            val file = files.optJSONObject(index) ?: continue
            if (file.optString("id") == fileId) {
                file.put("name", name)
                    .put("language", language)
                    .put("content", content)
                    .put("updated_at", now)
                found = true
                break
            }
        }
        require(found) { "Файл больше не существует." }
        project.put("updated_at", now)
        writeRoot(root)
    }

    fun deleteFile(projectId: String, fileId: String): Boolean {
        val root = readRoot()
        val project = findProject(root, projectId) ?: return false
        val source = project.optJSONArray("files") ?: JSONArray()
        val kept = JSONArray()
        var removed = false
        for (index in 0 until source.length()) {
            val file = source.optJSONObject(index) ?: continue
            if (file.optString("id") == fileId) {
                removed = true
            } else {
                kept.put(file)
            }
        }
        if (removed) {
            project.put("files", kept).put("updated_at", System.currentTimeMillis() / 1000L)
            writeRoot(root)
        }
        return removed
    }

    private fun findProject(root: JSONObject, projectId: String): JSONObject? {
        val projects = root.optJSONArray("projects") ?: JSONArray()
        for (index in 0 until projects.length()) {
            val project = projects.optJSONObject(index) ?: continue
            if (project.optString("id") == projectId) return project
        }
        return null
    }

    private fun normalizeName(raw: String, kind: String): String {
        val name = raw.trim()
        require(name.isNotBlank()) { "Укажи название." }
        if (kind == "file") {
            require(name.length <= 180) { "Путь файла не должен превышать 180 символов." }
            require(!name.startsWith("/") && !name.contains('\\\\') && !name.contains(':') && !name.any { it.code == 0 }) {
                "Недопустимый путь файла."
            }
            val parts = name.split("/")
            require(parts.size <= 5 && parts.none { it.isBlank() || it == "." || it == ".." }) {
                "Путь может содержать максимум пять безопасных уровней."
            }
            val allowedHiddenFiles = setOf(".gitignore", ".env.example")
            for (part in parts) {
                require(part.length <= 64 && (!part.startsWith(".") || part in allowedHiddenFiles)) {
                    "Недопустимое имя файла или скрытая директория."
                }
                require(part.matches(Regex("[\\p{L}\\p{N}._ -]+"))) {
                    "В имени файла разрешены буквы, цифры, пробел, точка, дефис и подчёркивание."
                }
            }
            require(parts.none { it.lowercase() in setOf(".git", ".idea", ".gradle", ".venv", "node_modules", "__pycache__") }) {
                "Служебные папки не сохраняются в рабочем пространстве."
            }
            return parts.joinToString("/")
        }
        require(name.length <= 64) { "Название не должно превышать 64 символа." }
        require(name != "." && name != "..") { "Недопустимое название." }
        require(name.matches(Regex("[\\p{L}\\p{N}._ -]+"))) {
            "Разрешены буквы, цифры, пробел, точка, дефис и подчёркивание."
        }
        return name
    }

    private fun readRoot(): JSONObject {
        val encrypted = preferences.getString(storageKey, null)
            ?: return JSONObject().put("projects", JSONArray())
        return try {
            val packed = Base64.decode(encrypted, Base64.NO_WRAP)
            require(packed.size > GCM_IV_BYTES) { "Повреждён файл хранилища." }
            val iv = packed.copyOfRange(0, GCM_IV_BYTES)
            val ciphertext = packed.copyOfRange(GCM_IV_BYTES, packed.size)
            val cipher = Cipher.getInstance("AES/GCM/NoPadding")
            cipher.init(Cipher.DECRYPT_MODE, secretKey(), GCMParameterSpec(GCM_TAG_BITS, iv))
            cipher.updateAAD(storageKey.toByteArray(Charsets.UTF_8))
            JSONObject(String(cipher.doFinal(ciphertext), Charsets.UTF_8))
        } catch (error: Exception) {
            throw IllegalStateException("Не удалось расшифровать локальные проекты. Данные не перезаписаны.", error)
        }
    }

    private fun writeRoot(root: JSONObject) {
        val plain = root.toString().toByteArray(Charsets.UTF_8)
        require(plain.size <= MAX_STORAGE_BYTES) { "Хранилище проектов заполнено." }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, secretKey())
        cipher.updateAAD(storageKey.toByteArray(Charsets.UTF_8))
        val encrypted = cipher.doFinal(plain)
        val packed = cipher.iv + encrypted
        preferences.edit()
            .putString(storageKey, Base64.encodeToString(packed, Base64.NO_WRAP))
            .apply()
    }

    private fun secretKey(): SecretKey {
        val keyStore = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val existing = keyStore.getKey(keyAlias, null) as? SecretKey
        if (existing != null) return existing

        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, "AndroidKeyStore")
        generator.init(
            KeyGenParameterSpec.Builder(
                keyAlias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return generator.generateKey()
    }

    private companion object {
        const val MAX_PROJECTS = 30
        const val MAX_FILES_PER_PROJECT = 80
        const val MAX_FILE_CHARS = 60000
        const val MAX_STORAGE_BYTES = 1_500_000
        const val GCM_IV_BYTES = 12
        const val GCM_TAG_BITS = 128
    }
}
