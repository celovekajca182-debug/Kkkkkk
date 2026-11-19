package com.swinai.app

import android.app.Activity
import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.ColorDrawable
import android.text.InputType
import android.text.method.PasswordTransformationMethod
import android.text.method.HideReturnsTransformationMethod
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import org.json.JSONObject
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.security.MessageDigest
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
    private lateinit var chatMessages: LinearLayout
    private lateinit var chatScroll: ScrollView
    private lateinit var codingPanel: LinearLayout
    private lateinit var chatInput: EditText
    private lateinit var sendButton: Button
    private lateinit var codeModeButton: Button
    private lateinit var languageSpinner: Spinner
    private lateinit var activitySwitch: Switch
    private lateinit var projectContextInput: EditText
    private lateinit var copyCodeButton: Button

    private var lastGeneratedCode: String = ""
    private var isRegisterMode = false
    private var isCodeMode = false
    private var showCodingActivity = false
    private var authToken: String? = null
    private var userEmail = ""
    private var sessionId = ""
    private var endpointValue: String = ""
    private var themeMode: String = "system"
    private var responseLanguage: String = "auto"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12, 14, 24)
        window.navigationBarColor = Color.rgb(12, 14, 24)

        val preferences = getPreferences(MODE_PRIVATE)
        endpointValue = preferences.getString("api_base", "https://kkkkkk-02vl.onrender.com") ?: "https://kkkkkk-02vl.onrender.com"
        themeMode = preferences.getString("theme_mode", "system") ?: "system"
        responseLanguage = preferences.getString("response_language", "auto") ?: "auto"
        authToken = SecureTokenStore.read(this)
        userEmail = preferences.getString("account_email", "") ?: ""
        if (authToken != null && userEmail.isNotBlank()) {
            sessionId = resolveAccountSessionId(userEmail)
            showChatScreen()
            validateSavedSession()
        } else {
            showAuthScreen()
        }
    }

    private fun resolveAccountSessionId(email: String): String {
        val normalizedEmail = email.trim().lowercase()
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(normalizedEmail.toByteArray(Charsets.UTF_8))
            .take(16)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        val stableSession = "account-$digest"
        val preferences = getPreferences(MODE_PRIVATE)
        val oldSession = preferences.getString("session_id:$email", null)

        // Migrate the previous device-specific backup without deleting the old copy.
        if (!oldSession.isNullOrBlank() && oldSession != stableSession) {
            val oldBackup = preferences.getString("chat_backup:$email:$oldSession", null)
            val newBackupKey = "chat_backup:$email:$stableSession"
            if (!oldBackup.isNullOrBlank() && preferences.getString(newBackupKey, null).isNullOrBlank()) {
                preferences.edit().putString(newBackupKey, oldBackup).apply()
            }
        }

        preferences.edit().putString("session_id:$email", stableSession).apply()
        return stableSession
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

    private fun authDp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    private fun authFieldBackground(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = authDp(12).toFloat()
        setColor(Color.rgb(27, 33, 39))
        setStroke(authDp(1), Color.rgb(55, 64, 72))
    }

    private fun authLabel(value: String): TextView = makeText(
        value,
        13f,
        Color.rgb(190, 199, 204)
    ).apply {
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun authInput(hintText: String, type: Int): EditText =
        makeEdit(hintText, type).apply {
            background = authFieldBackground()
            setPadding(authDp(16), authDp(8), authDp(16), authDp(8))
            minHeight = authDp(56)
            textSize = 16f
            typeface = Typeface.create("sans-serif", Typeface.NORMAL)
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_NEXT
        }

    private fun styleAuthButton(button: Button, primary: Boolean) {
        button.isAllCaps = false
        button.textSize = 15f
        button.typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        button.setPadding(authDp(16), 0, authDp(16), 0)
        button.background = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = authDp(13).toFloat()
            setColor(
                if (primary) Color.rgb(195, 231, 111)
                else Color.TRANSPARENT
            )
            if (!primary) setStroke(authDp(1), Color.rgb(58, 68, 75))
        }
        button.setTextColor(
            if (primary) Color.rgb(24, 30, 23)
            else Color.rgb(206, 215, 218)
        )
        button.stateListAnimator = null
        button.elevation = if (primary) authDp(2).toFloat() else 0f
    }

    private fun showAuthScreen(message: String? = null) {
        window.statusBarColor = Color.rgb(16, 21, 24)
        window.navigationBarColor = Color.rgb(16, 21, 24)

        val backgroundColor = Color.rgb(16, 21, 24)
        val accentColor = Color.rgb(195, 231, 111)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(backgroundColor)
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = false
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(authDp(24), authDp(28), authDp(24), authDp(26))
        }

        val brandRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val brandMark = TextView(this).apply {
            text = "S"
            textSize = 24f
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            gravity = Gravity.CENTER
            setTextColor(Color.rgb(23, 29, 22))
            background = GradientDrawable().apply {
                shape = GradientDrawable.RECTANGLE
                cornerRadius = authDp(12).toFloat()
                setColor(accentColor)
            }
        }
        brandRow.addView(brandMark, LinearLayout.LayoutParams(authDp(46), authDp(46)))
        val brandName = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(authDp(12), 0, 0, 0)
        }
        brandName.addView(makeText("SWIN AI", 18f, Color.WHITE).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        brandName.addView(makeText("ЛИЧНЫЙ КАБИНЕТ", 10f, Color.rgb(130, 143, 149)).apply {
            letterSpacing = 0.12f
        })
        brandRow.addView(brandName, LinearLayout.LayoutParams(
            0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f
        ))
        content.addView(brandRow)
        addGap(content, 42)

        content.addView(makeText(
            if (isRegisterMode) "НОВЫЙ АККАУНТ" else "С ВОЗВРАЩЕНИЕМ",
            11f,
            accentColor
        ).apply {
            letterSpacing = 0.14f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        addGap(content, 9)

        content.addView(makeText(
            if (isRegisterMode) "Создайте аккаунт" else "Войдите в SWIN AI",
            29f,
            Color.rgb(244, 247, 245)
        ).apply {
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
        })
        addGap(content, 8)
        content.addView(makeText(
            if (isRegisterMode)
                "Один аккаунт для работы с вашим помощником."
            else
                "Введите данные, чтобы продолжить работу.",
            14f,
            Color.rgb(151, 163, 168)
        ))
        addGap(content, 30)

        content.addView(authLabel("Электронная почта"))
        addGap(content, 8)
        emailInput = authInput(
            "name@example.com",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_EMAIL_ADDRESS
        ).apply {
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_NEXT
            setText(userEmail.takeIf { it.isNotBlank() } ?: "")
        }
        content.addView(emailInput, LinearLayout.LayoutParams(-1, authDp(56)))
        addGap(content, 18)

        content.addView(authLabel("Пароль"))
        addGap(content, 8)
        passwordInput = makeEdit(
            "Введите пароль",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        ).apply {
            background = ColorDrawable(Color.TRANSPARENT)
            setPadding(authDp(16), authDp(8), authDp(8), authDp(8))
            minHeight = authDp(54)
            textSize = 16f
            imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_NEXT
        }
        val passwordRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = authFieldBackground()
        }
        val revealPassword = TextView(this).apply {
            text = "Показать"
            textSize = 13f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(accentColor)
            gravity = Gravity.CENTER
            setPadding(authDp(8), 0, authDp(14), 0)
        }
        passwordRow.addView(passwordInput, LinearLayout.LayoutParams(
            0, authDp(56), 1f
        ))
        passwordRow.addView(revealPassword, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, authDp(56)
        ))
        var passwordVisible = false
        revealPassword.setOnClickListener {
            val cursor = passwordInput.selectionStart.coerceAtLeast(0)
            passwordVisible = !passwordVisible
            passwordInput.transformationMethod = if (passwordVisible) {
                HideReturnsTransformationMethod.getInstance()
            } else {
                PasswordTransformationMethod.getInstance()
            }
            revealPassword.text = if (passwordVisible) "Скрыть" else "Показать"
            passwordInput.setSelection(cursor.coerceAtMost(passwordInput.text.length))
        }
        content.addView(passwordRow, LinearLayout.LayoutParams(-1, authDp(56)))

        confirmInput = authInput(
            "Повторите пароль",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_PASSWORD
        )
        confirmInput.imeOptions = android.view.inputmethod.EditorInfo.IME_ACTION_DONE
        if (isRegisterMode) {
            addGap(content, 18)
            content.addView(authLabel("Повторите пароль"))
            addGap(content, 8)
            content.addView(confirmInput, LinearLayout.LayoutParams(-1, authDp(56)))
            addGap(content, 7)
            content.addView(makeText(
                "Используйте не менее 12 символов.",
                12f,
                Color.rgb(130, 143, 149)
            ))
        }

        addGap(content, 22)
        authStatus = makeText(
            message ?: if (isRegisterMode)
                "Пароль должен содержать минимум 12 символов."
            else
                "Ваши данные передаются по защищённому соединению.",
            12f,
            if (message.isNullOrBlank()) Color.rgb(130, 143, 149)
            else Color.rgb(235, 177, 135)
        )
        content.addView(authStatus)
        addGap(content, 12)

        authButton = makeButton(
            if (isRegisterMode) "Создать аккаунт" else "Войти"
        ) { submitAuthentication() }
        styleAuthButton(authButton, true)
        content.addView(authButton, LinearLayout.LayoutParams(-1, authDp(56)))
        addGap(content, 8)

        val switchButton = makeButton(
            if (isRegisterMode) "Уже есть аккаунт? Войти"
            else "Нет аккаунта? Зарегистрироваться"
        ) {
            isRegisterMode = !isRegisterMode
            showAuthScreen()
        }
        styleAuthButton(switchButton, false)
        content.addView(switchButton, LinearLayout.LayoutParams(-1, authDp(50)))

        addGap(content, 28)
        val serverDivider = View(this).apply {
            setBackgroundColor(Color.rgb(43, 52, 58))
        }
        content.addView(serverDivider, LinearLayout.LayoutParams(-1, authDp(1)))
        addGap(content, 15)

        val serverHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        serverHeader.addView(makeText(
            "Настройки подключения",
            13f,
            Color.rgb(174, 185, 190)
        ), LinearLayout.LayoutParams(0, authDp(38), 1f))
        val serverToggle = TextView(this).apply {
            text = "Изменить"
            textSize = 13f
            setTextColor(accentColor)
            gravity = Gravity.CENTER_VERTICAL
        }
        serverHeader.addView(serverToggle)
        content.addView(serverHeader)

        endpoint = makeEdit(
            "HTTPS-адрес сервера",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
        ).apply {
            setText(endpointValue)
            background = authFieldBackground()
            setPadding(authDp(14), authDp(8), authDp(14), authDp(8))
            minHeight = authDp(52)
            textSize = 13f
            visibility = View.GONE
        }
        content.addView(endpoint, LinearLayout.LayoutParams(-1, authDp(52)))
        serverToggle.setOnClickListener {
            val expanded = endpoint.visibility != View.VISIBLE
            endpoint.visibility = if (expanded) View.VISIBLE else View.GONE
            serverToggle.text = if (expanded) "Скрыть" else "Изменить"
        }

        addGap(content, 18)
        content.addView(makeText(
            "SWIN AI  •  Защищённое подключение",
            11f,
            Color.rgb(100, 113, 119)
        ).apply { gravity = Gravity.CENTER })

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
                        sessionId = resolveAccountSessionId(userEmail)
                        getPreferences(MODE_PRIVATE).edit()
                            .putString("account_email", userEmail)
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

    private fun isDarkTheme(): Boolean = when (themeMode) {
        "dark" -> true
        "light" -> false
        else -> (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) ==
            Configuration.UI_MODE_NIGHT_YES
    }

    private fun chatColor(key: String): Int {
        val dark = isDarkTheme()
        return when (key) {
            "background" -> if (dark) Color.rgb(20, 24, 22) else Color.rgb(250, 248, 243)
            "panel" -> if (dark) Color.rgb(31, 37, 33) else Color.rgb(255, 254, 251)
            "ink" -> if (dark) Color.rgb(235, 239, 233) else Color.rgb(43, 42, 38)
            "muted" -> if (dark) Color.rgb(157, 169, 159) else Color.rgb(126, 122, 113)
            "accent" -> if (dark) Color.rgb(190, 224, 132) else Color.rgb(190, 111, 82)
            "stroke" -> if (dark) Color.rgb(57, 68, 60) else Color.rgb(224, 219, 210)
            "assistantBubble" -> if (dark) Color.rgb(31, 37, 33) else Color.rgb(255, 254, 251)
            "userBubble" -> if (dark) Color.rgb(43, 55, 45) else Color.rgb(237, 230, 219)
            "codeBubble" -> if (dark) Color.rgb(23, 29, 25) else Color.rgb(243, 241, 235)
            "codePill" -> if (dark) Color.rgb(48, 39, 33) else Color.rgb(245, 237, 229)
            "composerHint" -> if (dark) Color.rgb(161, 171, 163) else Color.rgb(154, 149, 139)
            "sendText" -> if (dark) Color.rgb(28, 34, 28) else Color.WHITE
            "assistantLabel" -> if (dark) Color.rgb(222, 165, 133) else Color.rgb(177, 105, 79)
            "userLabel" -> if (dark) Color.rgb(191, 213, 156) else Color.rgb(139, 105, 84)
            else -> chatColor("ink")
        }
    }

    private fun showThemeSettings() {
        val options = arrayOf("Как в системе", "Светлая", "Тёмная")
        val selected = when (themeMode) {
            "light" -> 1
            "dark" -> 2
            else -> 0
        }
        AlertDialog.Builder(this)
            .setTitle("Оформление SWIN AI")
            .setSingleChoiceItems(options, selected) { dialog, which ->
                themeMode = when (which) {
                    1 -> "light"
                    2 -> "dark"
                    else -> "system"
                }
                getPreferences(MODE_PRIVATE).edit()
                    .putString("theme_mode", themeMode)
                    .apply()
                dialog.dismiss()
                showChatScreen()
            }
            .setNegativeButton("Закрыть", null)
            .show()
    }

    private fun chatDp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun chatBg(color: Int, radius: Int, stroke: Int? = null) = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = chatDp(radius).toFloat()
        setColor(color)
        if (stroke != null) setStroke(chatDp(1), stroke)
    }

    private fun copyMessageToClipboard(value: String, confirmation: String) {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(ClipData.newPlainText("SWIN AI", value))
        Toast.makeText(this, confirmation, Toast.LENGTH_SHORT).show()
    }

    private fun appendChatMessage(role: String, value: String, code: Boolean = false, copyText: String? = null): LinearLayout? {
        if (!::chatMessages.isInitialized) return null
        val user = role == "user"
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = if (user) Gravity.END else Gravity.START
            setPadding(0, chatDp(6), 0, chatDp(6))
        }
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(14), chatDp(11), chatDp(14), chatDp(11))
            background = if (user) chatBg(chatColor("userBubble"), 17)
                else chatBg(chatColor("assistantBubble"), 16, chatColor("stroke"))
        }
        val tag = makeText(if (user) "ВЫ" else "SWIN", 10f,
            if (user) chatColor("userLabel") else chatColor("assistantLabel")).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            letterSpacing = 0.12f
        }

        fun addBody(text: String, monospace: Boolean = false) {
            if (text.isBlank()) return
            val body = makeText(
                text,
                if (monospace) 13f else 15f,
                chatColor("ink")
            ).apply {
                setLineSpacing(chatDp(3).toFloat(), 1f)
                setTextIsSelectable(true)
                if (monospace) {
                    typeface = Typeface.create("monospace", Typeface.NORMAL)
                    background = chatBg(chatColor("codeBubble"), 9)
                    setPadding(chatDp(9), chatDp(9), chatDp(9), chatDp(9))
                }
            }
            card.addView(body, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = chatDp(4)
            })
        }

        fun addCopyAction(text: String, label: String, confirmation: String) {
            val action = TextView(this).apply {
                this.text = "⧉  $label"
                textSize = 12f
                setTextColor(chatColor("accent"))
                gravity = Gravity.END or Gravity.CENTER_VERTICAL
                setPadding(0, chatDp(8), chatDp(2), 0)
                setOnClickListener { copyMessageToClipboard(text, confirmation) }
            }
            card.addView(action, LinearLayout.LayoutParams(-1, chatDp(30)))
        }

        fun addCodeBlock(language: String, codeText: String) {
            val codeCard = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(chatDp(9), chatDp(7), chatDp(9), chatDp(9))
                background = chatBg(chatColor("codeBubble"), 10)
            }
            val toolbar = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            val languageLabel = makeText(
                language.ifBlank { "CODE" }.uppercase(),
                10f,
                chatColor("muted")
            ).apply {
                typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                letterSpacing = 0.08f
            }
            toolbar.addView(languageLabel, LinearLayout.LayoutParams(0, chatDp(28), 1f))
            val copy = TextView(this).apply {
                text = "⧉ Копировать"
                textSize = 11f
                setTextColor(chatColor("accent"))
                gravity = Gravity.CENTER
                setPadding(chatDp(7), 0, chatDp(2), 0)
                setOnClickListener {
                    copyMessageToClipboard(codeText, "Код скопирован")
                }
            }
            toolbar.addView(copy, LinearLayout.LayoutParams(-2, chatDp(28)))
            codeCard.addView(toolbar)
            val codeView = makeText(codeText, 13f, chatColor("ink")).apply {
                typeface = Typeface.create("monospace", Typeface.NORMAL)
                setTextIsSelectable(true)
                setLineSpacing(chatDp(2).toFloat(), 1f)
                setPadding(0, chatDp(5), 0, 0)
            }
            codeCard.addView(codeView)
            card.addView(codeCard, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT).apply {
                topMargin = chatDp(7)
            })
        }

        card.addView(tag)
        card.addView(View(this), LinearLayout.LayoutParams(1, chatDp(5)))

        if (code) {
            addBody(value, monospace = true)
            if (!copyText.isNullOrBlank()) {
                addCopyAction(copyText, "Копировать код", "Код скопирован")
            }
        } else if (!user) {
            val codeFence = Regex("(?s)```([A-Za-z0-9_+#.-]*)\\s*\\n(.*?)```")
            val matches = codeFence.findAll(value).toList()
            if (matches.isEmpty()) {
                addBody(value)
                addCopyAction(value, "Копировать ответ", "Ответ скопирован")
            } else {
                var cursor = 0
                for (match in matches) {
                    if (match.range.first > cursor) {
                        addBody(value.substring(cursor, match.range.first))
                    }
                    addCodeBlock(
                        match.groupValues[1],
                        match.groupValues[2].trimEnd()
                    )
                    cursor = match.range.last + 1
                }
                if (cursor < value.length) {
                    addBody(value.substring(cursor))
                }
                addCopyAction(value, "Копировать всё", "Ответ скопирован")
            }
        } else {
            addBody(value)
        }

        val maxW = (resources.displayMetrics.widthPixels * 0.88f).toInt()
        row.addView(card, LinearLayout.LayoutParams(maxW, ViewGroup.LayoutParams.WRAP_CONTENT))
        chatMessages.addView(row, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        chatScroll.post { chatScroll.fullScroll(View.FOCUS_DOWN) }
        return row
    }

    private fun showChatSettings() {
        val options = arrayOf(
            "Язык ответа",
            "Адрес сервера",
            "Очистить историю и память",
            "Выйти из аккаунта"
        )
        AlertDialog.Builder(this)
            .setTitle("Настройки SWIN AI")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showResponseLanguageSettings()
                    1 -> showEndpointSettings()
                    2 -> confirmClearChatMemory()
                    3 -> logout()
                }
            }
            .setNegativeButton("Закрыть", null)
            .show()
    }

    private fun showResponseLanguageSettings() {
        val codes = arrayOf("auto", "ru", "en", "de", "es", "fr", "zh", "uk")
        val labels = arrayOf(
            "Автоопределение по сообщению",
            "Русский",
            "English",
            "Deutsch",
            "Español",
            "Français",
            "中文",
            "Українська"
        )
        val selected = codes.indexOf(responseLanguage).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle("Язык ответа SWIN")
            .setSingleChoiceItems(labels, selected) { dialog, which ->
                responseLanguage = codes[which]
                getPreferences(MODE_PRIVATE).edit()
                    .putString("response_language", responseLanguage)
                    .apply()
                dialog.dismiss()
                Toast.makeText(
                    this,
                    "Язык ответа: " + labels[which],
                    Toast.LENGTH_SHORT
                ).show()
            }
            .setNegativeButton("Отмена", null)
            .show()
    }

    private fun showEndpointSettings() {
        val field = EditText(this).apply {
            setText(endpointValue)
            hint = "https://адрес-сервера"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(true)
        }
        AlertDialog.Builder(this)
            .setTitle("Подключение")
            .setMessage("Адрес API-сервера")
            .setView(field)
            .setPositiveButton("Сохранить") { _, _ ->
                val v = field.text.toString().trim().trimEnd('/')
                if (v.startsWith("https://")) {
                    endpointValue = v
                    endpoint.setText(v)
                    getPreferences(MODE_PRIVATE).edit().putString("api_base", v).apply()
                    Toast.makeText(this, "Адрес сохранён", Toast.LENGTH_SHORT).show()
                } else Toast.makeText(this, "Нужен HTTPS-адрес", Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton("Закрыть", null)
            .show()
    }

    private fun confirmClearChatMemory() {
        AlertDialog.Builder(this)
            .setTitle("Очистить память?")
            .setMessage("Будут удалены история этой сессии на сервере и резервная копия на телефоне.")
            .setPositiveButton("Очистить") { _, _ -> clearChatMemory() }
            .setNegativeButton("Отмена", null)
            .show()
    }

    private fun clearChatMemory() {
        val token = authToken
        thread {
            var serverCleared = false
            try {
                if (!token.isNullOrBlank()) {
                    requestJson(
                        "/memory/clear",
                        JSONObject().put("session_id", sessionId),
                        token
                    )
                    serverCleared = true
                }
            } catch (_: Exception) {
                // Local history is still cleared; server cleanup can be retried later.
            }
            getPreferences(MODE_PRIVATE).edit().remove(localHistoryKey()).apply()
            runOnUiThread {
                chatMessages.removeAllViews()
                appendChatMessage("assistant", "История очищена. Начнём заново.")
                val message = if (serverCleared) "История очищена на сервере и телефоне."
                    else "Копия на телефоне очищена. Сервер временно недоступен."
                Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showChatScreen() {
        val bg = chatColor("background")
        val ink = chatColor("ink")
        val muted = chatColor("muted")
        val accent = chatColor("accent")
        window.statusBarColor = bg
        window.navigationBarColor = bg
        window.decorView.systemUiVisibility = if (isDarkTheme()) 0 else
            (View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR)

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(16), chatDp(6), chatDp(16), chatDp(8))
            setBackgroundColor(bg)
        }
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val brand = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        brand.addView(makeText("SWIN AI", 19f, ink).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        brand.addView(makeText("Ваш AI-помощник", 11f, muted))
        header.addView(brand, LinearLayout.LayoutParams(0, chatDp(52), 1f))
        val codeMode = TextView(this).apply {
            text = if (isCodeMode) "Код ✓" else "Код"
            textSize = 13f
            gravity = Gravity.CENTER
            setTextColor(accent)
            setPadding(chatDp(14), 0, chatDp(14), 0)
            background = chatBg(chatColor("codePill"), 18)
            setOnClickListener {
                isCodeMode = !isCodeMode
                updateCodingControls()
                text = if (isCodeMode) "Код ✓" else "Код"
            }
        }
        val settings = TextView(this).apply {
            text = "⋯"
            textSize = 26f
            gravity = Gravity.CENTER
            setTextColor(ink)
            setOnClickListener { showChatSettings() }
        }
        val themeButton = TextView(this).apply {
            text = if (isDarkTheme()) "☀" else "☾"
            textSize = 21f
            gravity = Gravity.CENTER
            setTextColor(ink)
            contentDescription = "Сменить тему"
            setOnClickListener { showThemeSettings() }
        }
        header.addView(codeMode, LinearLayout.LayoutParams(-2, chatDp(40)))
        header.addView(themeButton, LinearLayout.LayoutParams(chatDp(40), chatDp(44)))
        header.addView(settings, LinearLayout.LayoutParams(chatDp(42), chatDp(44)))
        root.addView(header)

        endpoint = makeEdit("HTTPS-адрес сервера",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI).apply {
            setText(endpointValue)
            visibility = View.GONE
        }

        codingPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(10), chatDp(7), chatDp(10), chatDp(7))
            background = chatBg(chatColor("panel"), 14, chatColor("stroke"))
        }
        codeModeButton = makeButton("Параметры кода") {
            isCodeMode = !isCodeMode
            updateCodingControls()
        }
        languageSpinner = Spinner(this).apply {
            val langs = listOf("Автоопределение","Python","Godot 4 / GDScript","Kotlin","Java","JavaScript","TypeScript","C","C++","C#","Go","Rust","PHP","Ruby","Swift","Dart","SQL","Bash / Shell","PowerShell","Lua","R","Scala","Perl","Elixir","Haskell","HTML","CSS","XML","JSON","YAML","TOML","Dockerfile","GLSL / Shaders","Solidity","Другой язык")
            adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, langs)
        }
        projectContextInput = makeEdit("Код или структура проекта (необязательно)",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE, false).apply { maxLines = 3 }
        activitySwitch = Switch(this).apply {
            text = "Показывать этапы проверки"
            setTextColor(muted)
            setOnCheckedChangeListener { _, checked -> showCodingActivity = checked }
        }
        copyCodeButton = makeButton("Скопировать последний код") {
            val cb = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cb.setPrimaryClip(ClipData.newPlainText("SWIN AI code", lastGeneratedCode))
            Toast.makeText(this, "Код скопирован", Toast.LENGTH_SHORT).show()
        }.apply { isEnabled = false }
        codingPanel.addView(languageSpinner, LinearLayout.LayoutParams(-1, chatDp(44)))
        codingPanel.addView(projectContextInput, LinearLayout.LayoutParams(-1, -2))
        codingPanel.addView(activitySwitch, LinearLayout.LayoutParams(-1, chatDp(40)))
        codingPanel.addView(copyCodeButton, LinearLayout.LayoutParams(-1, chatDp(42)))
        root.addView(codingPanel, LinearLayout.LayoutParams(-1, -2).apply {
            topMargin = chatDp(5); bottomMargin = chatDp(4)
        })

        chatMessages = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, chatDp(10), 0, chatDp(16))
        }
        chatScroll = ScrollView(this).apply {
            isFillViewport = true
            isVerticalScrollBarEnabled = false
            addView(chatMessages, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(chatScroll, LinearLayout.LayoutParams(-1, 0, 1f))

        val composer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(chatDp(8), chatDp(6), chatDp(6), chatDp(6))
            background = chatBg(chatColor("panel"), 22, chatColor("stroke"))
            elevation = chatDp(2).toFloat()
        }
        val plus = TextView(this).apply {
            text = "+"
            textSize = 25f
            gravity = Gravity.CENTER
            setTextColor(muted)
            setOnClickListener {
                isCodeMode = !isCodeMode
                updateCodingControls()
                codeMode.text = if (isCodeMode) "Код ✓" else "Код"
            }
        }
        chatInput = makeEdit("Напишите сообщение…",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE, false).apply {
            background = ColorDrawable(Color.TRANSPARENT)
            setPadding(chatDp(8), chatDp(7), chatDp(8), chatDp(7))
            minHeight = chatDp(46)
            maxLines = 5
            textSize = 15f
            setTextColor(ink)
            setHintTextColor(chatColor("composerHint"))
        }
        sendButton = makeButton("↑") { sendMessage() }.apply {
            textSize = 22f
            setTextColor(chatColor("sendText"))
            background = chatBg(accent, 15)
            minWidth = chatDp(46)
            minimumWidth = chatDp(46)
            minHeight = chatDp(46)
            stateListAnimator = null
        }
        composer.addView(plus, LinearLayout.LayoutParams(chatDp(36), chatDp(46)))
        composer.addView(chatInput, LinearLayout.LayoutParams(0, -2, 1f))
        composer.addView(sendButton, LinearLayout.LayoutParams(chatDp(46), chatDp(46)))
        root.addView(composer, LinearLayout.LayoutParams(-1, -2).apply {
            topMargin = chatDp(8); bottomMargin = chatDp(5)
        })
        chatLog = makeText("", 14f)
        updateCodingControls()
        setContentView(root)
        loadChatHistory()
    }

    private fun localHistoryKey(): String = "chat_backup:$userEmail:$sessionId"

    private fun readLocalHistory(): JSONArray {
        val raw = getPreferences(MODE_PRIVATE).getString(localHistoryKey(), "[]") ?: "[]"
        return try {
            JSONArray(raw)
        } catch (_: Exception) {
            JSONArray()
        }
    }

    private fun cacheLocalMessage(role: String, content: String) {
        if (userEmail.isBlank() || sessionId.isBlank() || content.isBlank()) return
        val preferences = getPreferences(MODE_PRIVATE)
        val history = readLocalHistory()
        history.put(
            JSONObject()
                .put("role", role)
                .put("content", content.take(12000))
        )
        while (history.length() > 60) history.remove(0)
        preferences.edit().putString(localHistoryKey(), history.toString()).apply()
    }

    private fun loadChatHistory() {
        val token = authToken
        if (token.isNullOrBlank() || sessionId.isBlank()) {
            appendChatMessage("assistant", "Привет! Я SWIN AI. Чем могу помочь?")
            return
        }
        thread {
            try {
                val encodedSession = URLEncoder.encode(sessionId, "UTF-8")
                val response = requestJson(
                    "/memory/history?session_id=$encodedSession",
                    JSONObject(),
                    token,
                    method = "GET"
                )
                val serverMessages = response.optJSONArray("messages") ?: JSONArray()
                val localMessages = readLocalHistory()
                val restoreLocal = serverMessages.length() == 0 && localMessages.length() > 0
                val messages = if (restoreLocal) localMessages else serverMessages
                if (restoreLocal) {
                    runCatching {
                        requestJson(
                            "/memory/sync",
                            JSONObject()
                                .put("session_id", sessionId)
                                .put("messages", localMessages),
                            token
                        )
                    }
                }
                runOnUiThread {
                    chatMessages.removeAllViews()
                    if (messages.length() == 0) {
                        appendChatMessage("assistant", "Привет! Я SWIN AI. Чем могу помочь?")
                    } else {
                        for (index in 0 until messages.length()) {
                            val item = messages.optJSONObject(index) ?: continue
                            val role = item.optString("role", "assistant")
                            val content = item.optString("content", "")
                            if (content.isNotBlank()) {
                                val codeMarker = "\n\n[SWIN_CODE_BLOCK]\n"
                                val markerIndex = content.indexOf(codeMarker)
                                if (role == "assistant" && markerIndex >= 0) {
                                    val summary = content.substring(0, markerIndex).trim()
                                    val code = content.substring(markerIndex + codeMarker.length).trim()
                                    if (summary.isNotBlank()) appendChatMessage("assistant", summary)
                                    if (code.isNotBlank()) appendChatMessage("assistant", code, true, code)
                                } else {
                                    appendChatMessage(role, content)
                                }
                            }
                        }
                    }
                }
            } catch (_: Exception) {
                runOnUiThread {
                    if (chatMessages.childCount == 0) {
                        appendChatMessage("assistant", "Привет! Я SWIN AI. Чем могу помочь?")
                    }
                }
            }
        }
    }

    private fun updateCodingControls() {
        if (::codingPanel.isInitialized) codingPanel.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::languageSpinner.isInitialized) languageSpinner.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::projectContextInput.isInitialized) projectContextInput.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::activitySwitch.isInitialized) activitySwitch.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::copyCodeButton.isInitialized) copyCodeButton.visibility = if (isCodeMode) View.VISIBLE else View.GONE
        if (::sendButton.isInitialized) {
            sendButton.text = "➤"
            sendButton.contentDescription = if (isCodeMode) "Создать код" else "Отправить"
        }
        if (::chatInput.isInitialized) chatInput.hint =
            if (isCodeMode) "Опишите задачу для кода…" else "Напишите сообщение…"
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
            appendChatMessage("assistant", "Укажи HTTPS-адрес API-сервера в настройках.")
            return
        }
        endpointValue = base
        getPreferences(MODE_PRIVATE).edit().putString("api_base", base).apply()

        val codingLanguageValues = listOf(
            "auto", "python", "gdscript", "kotlin",
            "java", "javascript", "typescript", "c", "cpp", "csharp",
            "go", "rust", "php", "ruby", "swift", "dart", "sql",
            "bash", "powershell", "lua", "r", "scala", "perl",
            "elixir", "haskell", "html", "css", "xml", "json", "yaml",
            "toml", "dockerfile", "glsl", "solidity", "other"
        )
        val selectedCodingLanguage = codingLanguageValues.getOrElse(languageSpinner.selectedItemPosition) { "auto" }
        val includeCodingActivity = showCodingActivity
        val selectedProjectContext = projectContextInput.text.toString().take(4000)
        val requestCodeMode = isCodeMode
        val selectedResponseLanguage = responseLanguage

        chatInput.text.clear()
        appendChatMessage("user", message)
        cacheLocalMessage("user", message)
        val pendingReply = appendChatMessage(
            "assistant",
            if (requestCodeMode) "Готовлю результат…" else "Думаю…"
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
                        .put("session_id", sessionId)
                    val response = requestJson("/code/assist", payload, token)
                    val output = StringBuilder()
                    output.append("\nSWIN AI • РЕЖИМ РАЗРАБОТКИ\n")
                    output.append(response.optString("summary", "Результат готов."))
                    output.append("\nСтатус: ")
                    output.append(response.optString("status", "unknown"))
                    val validation = response.optJSONObject("validation")
                    if (validation != null) {
                        output.append("\nПроверка синтаксиса: ")
                        output.append(validation.optString("message", "не выполнена"))
                    }
                    val verification = response.optJSONObject("verification")
                    if (verification != null) {
                        output.append("\n\nПЕРЕПРОВЕРКА:")
                        val checkNames = listOf("research", "syntax", "cross_check", "runtime_tests")
                        val checkLabels = mapOf(
                            "research" to "Поиск в интернете",
                            "syntax" to "Статический анализ",
                            "cross_check" to "Сверка источников",
                            "runtime_tests" to "Запуск тестов"
                        )
                        for (key in checkNames) {
                            val check = verification.optJSONObject(key) ?: continue
                            output.append("\n• ")
                            output.append(checkLabels[key] ?: key)
                            output.append(" — ")
                            output.append(check.optString("status", "unknown"))
                            val detail = check.optString("message", "")
                            if (detail.isNotBlank()) {
                                output.append(": ")
                                output.append(detail)
                            }
                        }
                    }
                    val code = response.optString("code", "")
                    if (code.isNotBlank()) {
                        output.append("\n\nКОД:\n")
                        output.append(code)
                        output.append("\n")
                    } else {
                        output.append("\nНе удалось подготовить код, который прошёл проверку. Уточни задачу или попробуй снова.\n")
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
                    }
                    // Research sources remain visible even when the detailed activity log is hidden.
                    val refs = response.optJSONArray("references")
                    if (refs != null && refs.length() > 0) {
                        output.append("\nИсточники документации:\n")
                        for (i in 0 until refs.length()) {
                            val item = refs.optJSONObject(i) ?: continue
                            output.append("• ")
                            output.append(item.optString("source_type", "Веб-источник"))
                            output.append(": ")
                            output.append(item.optString("title"))
                            output.append(" — ")
                            output.append(item.optString("url"))
                            output.append("\n")
                        }
                    }
                    val elapsed = response.optDouble("elapsed_ms", -1.0)
                    if (elapsed >= 0) {
                        output.append("\nВремя обработки: ")
                        output.append(elapsed)
                        output.append(" мс")
                    }
                    val warnings = response.optJSONArray("warnings")
                    if (warnings != null) {
                        for (i in 0 until warnings.length()) {
                            output.append("\n⚠ ")
                            output.append(warnings.optString(i))
                        }
                    }
                    runOnUiThread {
                        pendingReply?.let { chatMessages.removeView(it) }
                        if (code.isNotBlank()) {
                            lastGeneratedCode = code
                            copyCodeButton.isEnabled = true
                        }
                        if (code.isNotBlank()) {
                            val summary = output.toString().substringBefore("\n\nКОД:\n").trim()
                            appendChatMessage("assistant", summary)
                            appendChatMessage("assistant", code, true, code)
                            cacheLocalMessage("assistant", summary + "\n\n[SWIN_CODE_BLOCK]\n" + code)
                        } else {
                            val outputText = output.toString()
                            appendChatMessage("assistant", outputText)
                            cacheLocalMessage("assistant", outputText)
                        }
                        sendButton.isEnabled = true
                    }
                } else {
                    val response = requestJson(
                        "/chat",
                        JSONObject()
                            .put("message", message)
                            .put("session_id", sessionId)
                            .put("response_language", selectedResponseLanguage),
                        token
                    )
                    val answer = response.optString("answer", "Сервер вернул пустой ответ.")
                    val elapsed = response.optDouble("generation_ms", -1.0)
                    val mode = response.optString("mode", "unknown")
                    val language = response.optString("response_language", selectedResponseLanguage)
                    val meta = if (elapsed >= 0) "\n[$mode • $language • $elapsed мс]" else "\n[$mode • $language]"
                    runOnUiThread {
                        pendingReply?.let { chatMessages.removeView(it) }
                        appendChatMessage("assistant", "$answer$meta")
                        cacheLocalMessage("assistant", "$answer$meta")
                        sendButton.isEnabled = true
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    pendingReply?.let { chatMessages.removeView(it) }
                    if (error is ApiException && error.statusCode == 401) {
                        clearLocalSession("Сессия истекла. Войди снова.")
                    } else {
                        appendChatMessage("assistant", "Ошибка: ${error.message}")
                        cacheLocalMessage("assistant", "Ошибка: ${error.message}")
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
            connection.connectTimeout = 30000
            connection.readTimeout = 120000
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
