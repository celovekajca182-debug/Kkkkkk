package com.swinai.app

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import kotlin.concurrent.thread

private class ApiException(val statusCode: Int, message: String) : Exception(message)

class MainActivity : Activity() {
    private lateinit var endpoint: EditText
    private lateinit var emailInput: EditText
    private lateinit var passwordInput: EditText
    private lateinit var confirmInput: EditText
    private lateinit var authStatus: TextView
    private lateinit var authButton: Button
    private lateinit var chatLog: TextView
    private lateinit var chatInput: EditText
    private lateinit var sendButton: Button
    private lateinit var codeModeButton: Button
    private lateinit var languageSpinner: Spinner
    private lateinit var activitySwitch: Switch
    private lateinit var projectContextInput: EditText

    private var isRegisterMode = false
    private var isCodeMode = false
    private var showCodingActivity = false
    private var authToken: String? = null
    private var userEmail = ""
    private var sessionId = ""
    private var endpointValue: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12, 14, 24)
        window.navigationBarColor = Color.rgb(12, 14, 24)

        val preferences = getPreferences(MODE_PRIVATE)
        endpointValue = preferences.getString("api_base", "") ?: ""
        authToken = SecureTokenStore.read(this)
        userEmail = preferences.getString("account_email", "") ?: ""
        if (authToken != null && userEmail.isNotBlank()) {
            sessionId = preferences.getString("session_id:$userEmail", null)
                ?: UUID.randomUUID().toString()
            preferences.edit().putString("session_id:$userEmail", sessionId).apply()
            showChatScreen()
            validateSavedSession()
        } else {
            showAuthScreen()
        }
    }

    private fun darkRoot(): LinearLayout = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        setPadding(22, 20, 22, 18)
        setBackgroundColor(Color.rgb(12, 14, 24))
    }

    private fun makeText(value: String, size: Float, color: Int = Color.WHITE): TextView =
        TextView(this).apply {
            text = value
            textSize = size
            setTextColor(color)
        }

    private fun makeEdit(
        hintText: String,
        type: Int,
        singleLine: Boolean = true
    ): EditText = EditText(this).apply {
        hint = hintText
        inputType = type
        setSingleLine(singleLine)
        textSize = 16f
        setTextColor(Color.WHITE)
        setHintTextColor(Color.rgb(145, 150, 165))
        setPadding(14, 10, 14, 10)
        minHeight = 50
    }

    private fun makeButton(label: String, onClick: () -> Unit): Button =
        Button(this).apply {
            text = label
            isAllCaps = false
            textSize = 15f
            setOnClickListener { onClick() }
        }

    private fun addGap(root: LinearLayout, height: Int = 10) {
        root.addView(View(this), LinearLayout.LayoutParams(1, height))
    }

    private fun showAuthScreen(message: String? = null) {
        val root = darkRoot()
        val scroll = ScrollView(this)
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(4, 18, 4, 18)
        }

        val title = makeText("SWIN AI", 30f, Color.rgb(170, 145, 255)).apply {
            gravity = Gravity.CENTER
        }
        val subtitle = makeText(
            if (isRegisterMode) "Создай аккаунт по электронной почте" else "Войди в свой аккаунт",
            15f,
            Color.LTGRAY
        ).apply { gravity = Gravity.CENTER }

        endpoint = makeEdit(
            "HTTPS-адрес сервера SWIN AI",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        ).apply { setText(endpointValue) }
        emailInput = makeEdit(
            "Электронная почта",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
        )
        passwordInput = makeEdit(
            "Пароль (минимум 12 символов)",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        )
        confirmInput = makeEdit(
            "Повтори пароль",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        ).apply {
            visibility = if (isRegisterMode) View.VISIBLE else View.GONE
        }

        authStatus = makeText(message ?: "Пароль хранится на сервере только в виде хеша.", 13f, Color.LTGRAY)
        authButton = makeButton(
            if (isRegisterMode) "Зарегистрироваться" else "Войти"
        ) { submitAuthentication() }
        val switchButton = makeButton(
            if (isRegisterMode) "У меня уже есть аккаунт — войти" else "Создать новый аккаунт"
        ) {
            isRegisterMode = !isRegisterMode
            showAuthScreen()
        }

        content.addView(title, LinearLayout.LayoutParams(-1, 58))
        content.addView(subtitle, LinearLayout.LayoutParams(-1, 36))
        addGap(content, 12)
        content.addView(endpoint, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        addGap(content, 8)
        content.addView(emailInput, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        addGap(content, 8)
        content.addView(passwordInput, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        addGap(content, 8)
        content.addView(confirmInput, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        addGap(content, 14)
        content.addView(authStatus, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        addGap(content, 8)
        content.addView(authButton, LinearLayout.LayoutParams(-1, 54))
        content.addView(switchButton, LinearLayout.LayoutParams(-1, 50))
        scroll.addView(content)
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        setContentView(root)
    }

    private fun submitAuthentication() {
        val base = endpoint.text.toString().trim().trimEnd('/')
        val email = emailInput.text.toString().trim()
        val password = passwordInput.text.toString()
        if (!base.startsWith("https://")) {
            authStatus.text = "Укажи HTTPS-адрес своего API-сервера."
            return
        }
        if (email.isBlank() || password.isBlank()) {
            authStatus.text = "Заполни почту и пароль."
            return
        }
        if (isRegisterMode) {
            if (password.length < 12) {
                authStatus.text = "Пароль должен содержать минимум 12 символов."
                return
            }
            if (password != confirmInput.text.toString()) {
                authStatus.text = "Пароли не совпадают."
                return
            }
        }

        endpointValue = base
        getPreferences(MODE_PRIVATE).edit().putString("api_base", base).apply()
        authButton.isEnabled = false
        authStatus.text = if (isRegisterMode) "Создаю аккаунт..." else "Проверяю данные..."

        val route = if (isRegisterMode) "/auth/register" else "/auth/login"
        thread {
            try {
                val response = requestJson(
                    route,
                    JSONObject().put("email", email).put("password", password)
                )
                val token = response.getString("token")
                val user = response.getJSONObject("user")
                runOnUiThread {
                    try {
                        SecureTokenStore.save(this, token)
                        authToken = token
                        userEmail = user.optString("email", email)
                        sessionId = getPreferences(MODE_PRIVATE)
                            .getString("session_id:$userEmail", null)
                            ?: UUID.randomUUID().toString()
                        getPreferences(MODE_PRIVATE).edit()
                            .putString("account_email", userEmail)
                            .putString("session_id:$userEmail", sessionId)
                            .apply()
                        showChatScreen()
                    } catch (error: Exception) {
                        authButton.isEnabled = true
                        authStatus.text = "Не удалось безопасно сохранить сессию: \${error.message}"
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    authButton.isEnabled = true
                    authStatus.text = error.message ?: "Не удалось подключиться к серверу."
                }
            }
        }
    }

    private fun showChatScreen() {
        val root = darkRoot()
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val account = makeText("SWIN AI  •  $userEmail", 15f, Color.rgb(190, 175, 255))
        val logout = makeButton("Выйти") { logout() }
        header.addView(account, LinearLayout.LayoutParams(0, 52, 1f))
        header.addView(logout, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            48
        ))

        val codingControls = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        codeModeButton = makeButton("Режим: Чат") {
            isCodeMode = !isCodeMode
            updateCodingControls()
        }
        languageSpinner = Spinner(this).apply {
            val languages = listOf(
                "Автоопределение",
                "Python",
                "Godot 4 / GDScript",
                "Kotlin / Android",
                "JavaScript",
                "TypeScript",
                "HTML",
                "CSS",
                "JSON"
            )
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                languages
            )
        }
        projectContextInput = makeEdit(
            "Текущий код или структура проекта (необязательно)",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE,
            singleLine = false
        ).apply { maxLines = 4 }
        activitySwitch = Switch(this).apply {
            text = "Показывать этапы поиска и проверки"
            setTextColor(Color.LTGRAY)
            isChecked = false
            setOnCheckedChangeListener { _, checked ->
                showCodingActivity = checked
            }
        }
        codingControls.addView(codeModeButton, LinearLayout.LayoutParams(-1, 48))
        codingControls.addView(languageSpinner, LinearLayout.LayoutParams(-1, 48))
        codingControls.addView(projectContextInput, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        codingControls.addView(activitySwitch, LinearLayout.LayoutParams(-1, 44))
        root.addView(codingControls, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))

        endpoint = makeEdit(
            "HTTPS-адрес сервера",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        ).apply { setText(endpointValue) }

        chatLog = makeText(
            "SWIN AI 0.5\nАккаунт: $userEmail\nСессия защищена. Напиши сообщение.\n",
            16f
        ).apply { setPadding(8, 14, 8, 14) }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            addView(chatLog)
        }

        chatInput = makeEdit(
            "Напиши сообщение...",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE,
            singleLine = false
        ).apply { maxLines = 4 }
        sendButton = makeButton("Отправить") { sendMessage() }

        root.addView(header, LinearLayout.LayoutParams(-1, 54))
        root.addView(endpoint, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(chatInput, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(sendButton, LinearLayout.LayoutParams(-1, 52))
        updateCodingControls()
        setContentView(root)
    }

    private fun updateCodingControls() {
        codeModeButton.text = if (isCodeMode) "Режим: Разработка кода" else "Режим: Чат"
        languageSpinner.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        projectContextInput.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        activitySwitch.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::sendButton.isInitialized) {
            sendButton.text = if (isCodeMode) "Исследовать и создать код" else "Отправить"
        }
        if (::chatInput.isInitialized) {
            chatInput.hint = if (isCodeMode) {
                "Опиши задачу и что должен делать код..."
            } else {
                "Напиши сообщение..."
            }
        }
    }

    private fun validateSavedSession() {
        val token = authToken ?: return
        thread {
            try {
                requestJson("/auth/me", JSONObject(), token, method = "GET")
            } catch (error: Exception) {
                if (error is ApiException && error.statusCode == 401) {
                    runOnUiThread { clearLocalSession("Сессия истекла. Войди снова.") }
                }
            }
        }
    }

    private fun sendMessage() {
        val message = chatInput.text.toString().trim()
        if (message.isEmpty()) return
        val token = authToken
        if (token.isNullOrBlank()) {
            clearLocalSession("Сначала войди в аккаунт.")
            return
        }

        val base = endpoint.text.toString().trim().trimEnd('/')
        if (!base.startsWith("https://")) {
            chatLog.append("\nУкажи HTTPS-адрес API-сервера.\n")
            return
        }
        endpointValue = base
        getPreferences(MODE_PRIVATE).edit().putString("api_base", base).apply()

        val codingLanguageValues = listOf(
            "auto", "python", "gdscript", "kotlin",
            "javascript", "typescript", "html", "css", "json"
        )
        val selectedCodingLanguage = codingLanguageValues.getOrElse(languageSpinner.selectedItemPosition) { "auto" }
        val includeCodingActivity = showCodingActivity
        val selectedProjectContext = projectContextInput.text.toString().take(4000)
        val requestCodeMode = isCodeMode

        chatInput.text.clear()
        chatLog.append(
            if (requestCodeMode) "\nТы • задача: $message\nSWIN AI: изучаю документацию и готовлю код...\n"
            else "\nТы: $message\nSWIN AI: думаю...\n"
        )
        sendButton.isEnabled = false

        thread {
            try {
                if (requestCodeMode) {
                    val payload = JSONObject()
                        .put("task", message)
                        .put("language", selectedCodingLanguage)
                        .put("research_web", true)
                        .put("show_activity", includeCodingActivity)
                        .put("project_context", selectedProjectContext)
                    val response = requestJson("/code/assist", payload, token)
                    val output = StringBuilder()
                    output.append("\nSWIN AI • РЕЖИМ РАЗРАБОТКИ\n")
                    output.append(response.optString("summary", "Результат готов."))
                    output.append("\nСтатус: ")
                    output.append(response.optString("status", "unknown"))
                    val validation = response.optJSONObject("validation")
                    if (validation != null) {
                        output.append("\nПроверка: ")
                        output.append(validation.optString("message", "не выполнена"))
                    }
                    val code = response.optString("code", "")
                    if (code.isNotBlank()) {
                        output.append("\n\nКОД:\n")
                        output.append(code)
                        output.append("\n")
                    } else {
                        output.append("\nКод не создан: проверь, загружены ли веса модели на сервер.\n")
                    }

                    if (includeCodingActivity) {
                        val plan = response.optJSONArray("plan")
                        if (plan != null && plan.length() > 0) {
                            output.append("\nПлан работы:\n")
                            for (i in 0 until plan.length()) {
                                output.append("• ")
                                output.append(plan.optString(i))
                                output.append("\n")
                            }
                        }
                        val activity = response.optJSONArray("activity")
                        if (activity != null && activity.length() > 0) {
                            output.append("\nЭтапы:\n")
                            for (i in 0 until activity.length()) {
                                val item = activity.optJSONObject(i) ?: continue
                                output.append("• ")
                                output.append(item.optString("label"))
                                output.append(" [")
                                output.append(item.optString("status"))
                                output.append("]\n")
                            }
                        }
                        val refs = response.optJSONArray("references")
                        if (refs != null && refs.length() > 0) {
                            output.append("\nИсточники:\n")
                            for (i in 0 until refs.length()) {
                                val item = refs.optJSONObject(i) ?: continue
                                output.append("• ")
                                output.append(item.optString("title"))
                                output.append(" — ")
                                output.append(item.optString("url"))
                                output.append("\n")
                            }
                        }
                    }
                    val warnings = response.optJSONArray("warnings")
                    if (warnings != null) {
                        for (i in 0 until warnings.length()) {
                            output.append("\n⚠ ")
                            output.append(warnings.optString(i))
                        }
                    }
                    runOnUiThread {
                        chatLog.append(output.toString() + "\n")
                        sendButton.isEnabled = true
                    }
                } else {
                    val response = requestJson(
                        "/chat",
                        JSONObject().put("message", message).put("session_id", sessionId),
                        token
                    )
                    val answer = response.optString("answer", "Сервер вернул пустой ответ.")
                    val elapsed = response.optDouble("generation_ms", -1.0)
                    val mode = response.optString("mode", "unknown")
                    val meta = if (elapsed >= 0) "\n[$mode • $elapsed мс]" else "\n[$mode]"
                    runOnUiThread {
                        chatLog.append("SWIN AI: $answer$meta\n")
                        sendButton.isEnabled = true
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    if (error is ApiException && error.statusCode == 401) {
                        clearLocalSession("Сессия истекла. Войди снова.")
                    } else {
                        chatLog.append("Ошибка: ${error.message}\n")
                        sendButton.isEnabled = true
                    }
                }
            }
        }
    }

    private fun logout() {
        val token = authToken
        if (!token.isNullOrBlank()) {
            thread {
                try {
                    requestJson("/auth/logout", JSONObject(), token)
                } catch (_: Exception) {
                    // Local sign-out still clears the token if the server is unreachable.
                }
                runOnUiThread { clearLocalSession("Ты вышел из аккаунта.") }
            }
        } else {
            clearLocalSession("Ты вышел из аккаунта.")
        }
    }

    private fun clearLocalSession(message: String) {
        SecureTokenStore.clear(this)
        authToken = null
        userEmail = ""
        sessionId = ""
        showAuthScreen(message)
    }

    private fun requestJson(
        path: String,
        payload: JSONObject,
        token: String? = null,
        method: String = "POST"
    ): JSONObject {
        val base = endpointValue.ifBlank {
            getPreferences(MODE_PRIVATE).getString("api_base", "") ?: ""
        }.trimEnd('/')
        if (!base.startsWith("https://")) {
            throw IllegalStateException("Нужен HTTPS-адрес сервера.")
        }

        val connection = URL(base + path).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = method
            connection.connectTimeout = 12000
            connection.readTimeout = 90000
            connection.setRequestProperty("Accept", "application/json")
            if (!token.isNullOrBlank()) {
                connection.setRequestProperty("Authorization", "Bearer $token")
            }
            if (method != "GET") {
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                connection.outputStream.use {
                    it.write(payload.toString().toByteArray(Charsets.UTF_8))
                }
            }
            val code = connection.responseCode
            val stream = if (code in 200..299) connection.inputStream else connection.errorStream
            val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: "{}"
            val json = try {
                JSONObject(body)
            } catch (_: Exception) {
                JSONObject().put("detail", body.ifBlank { "HTTP $code" })
            }
            if (code !in 200..299) {
                throw ApiException(code, json.optString("detail", "Ошибка сервера HTTP $code"))
            }
            return json
        } finally {
            connection.disconnect()
        }
    }
}
