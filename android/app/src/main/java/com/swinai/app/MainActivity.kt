package com.swinai.app

import android.app.Activity
import android.app.AlertDialog
import android.app.ProgressDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Base64
import android.view.ContextThemeWrapper
import android.content.res.Configuration
import android.content.res.ColorStateList
import android.os.Bundle
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.ColorDrawable
import android.text.InputType
import android.text.util.Linkify
import android.text.method.LinkMovementMethod
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
import java.io.ByteArrayOutputStream
import java.io.ByteArrayInputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream
import java.security.MessageDigest
import kotlin.concurrent.thread

private class ApiException(val statusCode: Int, message: String) : Exception(message)

class MainActivity : Activity() {
    private companion object {
        const val REQUEST_PICK_IMAGES = 7401
        const val REQUEST_IMPORT_PROJECT = 7402
        const val REQUEST_EXPORT_PROJECT = 7403
    }

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
    private lateinit var attachmentPreview: LinearLayout
    private lateinit var attachmentScroll: HorizontalScrollView
    private val pendingImageAttachments = mutableListOf<ChatImageAttachment>()
    private var pendingProjectExportId: String? = null
    private data class ChatImageAttachment(
        val dataUrl: String,
        val thumbnail: Bitmap,
        val displayName: String
    )
    private var welcomePanel: View? = null
    private var codeModeHeader: TextView? = null
    private var memoryDialog: AlertDialog? = null

    private var lastGeneratedCode: String = ""
    private var isRegisterMode = false
    private var isCodeMode = false
    private var showCodingActivity = false
    private var authToken: String? = null
    private var userEmail = ""
    private var sessionId = ""
    private var endpointValue: String = ""
    private var themeMode: String = "system"
    private var responseLanguage: String = "ru"

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != Activity.RESULT_OK || data == null) return
        when (requestCode) {
            REQUEST_PICK_IMAGES -> handlePickedImages(data)
            REQUEST_IMPORT_PROJECT -> data.data?.let { importWorkspaceArchive(it) }
            REQUEST_EXPORT_PROJECT -> {
                val target = data.data
                val projectId = pendingProjectExportId
                pendingProjectExportId = null
                if (target != null && !projectId.isNullOrBlank()) {
                    exportWorkspaceArchive(projectId, target)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = chatColor("background")
        window.navigationBarColor = chatColor("background")

        val preferences = getPreferences(MODE_PRIVATE)
        endpointValue = preferences.getString("api_base", "https://kkkkkk-02vl.onrender.com") ?: "https://kkkkkk-02vl.onrender.com"
        themeMode = preferences.getString("theme_mode", "system") ?: "system"
        // One-time migration: replace the former auto-detect default with Russian.
        // A language explicitly selected by the user is preserved.
        val languageDefaultMigrated = preferences.getBoolean("russian_default_v1", false)
        val savedResponseLanguage = preferences.getString("response_language", null)
        responseLanguage = if (!languageDefaultMigrated) {
            val initialLanguage = savedResponseLanguage
                ?.takeIf { it.isNotBlank() && it != "auto" } ?: "ru"
            preferences.edit()
                .putString("response_language", initialLanguage)
                .putBoolean("russian_default_v1", true)
                .apply()
            initialLanguage
        } else {
            preferences.getString("response_language", "ru") ?: "ru"
        }
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

        val activeKey = "active_session_id:$normalizedEmail"
        val activeSession = preferences.getString(activeKey, null)
            ?.takeIf { it.isNotBlank() } ?: stableSession
        preferences.edit()
            .putString("session_id:$email", stableSession)
            .putString(activeKey, activeSession)
            .apply()
        return activeSession
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
        setColor(chatColor("panel"))
        setStroke(authDp(1), chatColor("stroke"))
    }

    private fun authLabel(value: String): TextView = makeText(
        value,
        13f,
        chatColor("muted")
    ).apply {
        typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
    }

    private fun authInput(hintText: String, type: Int): EditText =
        makeEdit(hintText, type).apply {
            background = authFieldBackground()
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
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
                if (primary) chatColor("accent")
                else Color.TRANSPARENT
            )
            if (!primary) setStroke(authDp(1), chatColor("stroke"))
        }
        button.setTextColor(
            if (primary) chatColor("sendText")
            else chatColor("ink")
        )
        button.stateListAnimator = null
        button.elevation = if (primary) authDp(2).toFloat() else 0f
    }

    private fun showAuthScreen(message: String? = null) {
        val backgroundColor = chatColor("background")
        val accentColor = chatColor("accent")
        val inkColor = chatColor("ink")
        val mutedColor = chatColor("muted")
        window.statusBarColor = backgroundColor
        window.navigationBarColor = backgroundColor
        window.decorView.systemUiVisibility = if (isDarkTheme()) 0 else
            (View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR or View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR)

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
            setTextColor(chatColor("sendText"))
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
        brandName.addView(makeText("SWIN AI", 18f, inkColor).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        brandName.addView(makeText("ЛИЧНЫЙ КАБИНЕТ", 10f, mutedColor).apply {
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
            inkColor
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
            mutedColor
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
            setTextColor(inkColor)
            setHintTextColor(mutedColor)
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
                mutedColor
            ))
        }

        addGap(content, 22)
        authStatus = makeText(
            message ?: if (isRegisterMode)
                "Пароль должен содержать минимум 12 символов."
            else
                "Ваши данные передаются по защищённому соединению.",
            12f,
            if (message.isNullOrBlank()) mutedColor
            else chatColor("danger")
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
            setBackgroundColor(chatColor("stroke"))
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
            mutedColor
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
            mutedColor
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
            "background" -> if (dark) Color.rgb(18, 19, 22) else Color.rgb(247, 245, 241)
            "panel" -> if (dark) Color.rgb(28, 29, 34) else Color.rgb(255, 255, 253)
            "ink" -> if (dark) Color.rgb(239, 238, 236) else Color.rgb(39, 38, 35)
            "muted" -> if (dark) Color.rgb(158, 158, 164) else Color.rgb(119, 115, 107)
            "accent" -> if (dark) Color.rgb(226, 178, 125) else Color.rgb(156, 101, 55)
            "stroke" -> if (dark) Color.rgb(55, 56, 63) else Color.rgb(226, 222, 214)
            "assistantBubble" -> if (dark) Color.rgb(28, 29, 34) else Color.rgb(255, 255, 253)
            "userBubble" -> if (dark) Color.rgb(48, 42, 37) else Color.rgb(246, 238, 227)
            "codeBubble" -> if (dark) Color.rgb(20, 21, 25) else Color.rgb(243, 241, 237)
            "codePill" -> if (dark) Color.rgb(49, 41, 35) else Color.rgb(247, 239, 229)
            "composerHint" -> if (dark) Color.rgb(153, 153, 160) else Color.rgb(139, 133, 124)
            "sendText" -> if (dark) Color.rgb(31, 26, 22) else Color.WHITE
            "assistantLabel" -> if (dark) Color.rgb(225, 184, 142) else Color.rgb(149, 99, 60)
            "userLabel" -> if (dark) Color.rgb(213, 177, 142) else Color.rgb(126, 86, 54)
            "success" -> if (dark) Color.rgb(151, 202, 157) else Color.rgb(61, 129, 78)
            "danger" -> if (dark) Color.rgb(235, 135, 126) else Color.rgb(190, 67, 61)
            "syntaxKeyword" -> if (dark) Color.rgb(196, 167, 255) else Color.rgb(112, 75, 174)
            "syntaxString" -> if (dark) Color.rgb(174, 215, 145) else Color.rgb(48, 125, 75)
            "syntaxComment" -> if (dark) Color.rgb(132, 147, 137) else Color.rgb(126, 137, 128)
            "syntaxNumber" -> if (dark) Color.rgb(242, 190, 130) else Color.rgb(168, 96, 40)
            "syntaxFunction" -> if (dark) Color.rgb(133, 194, 232) else Color.rgb(38, 111, 156)
            "syntaxTag" -> if (dark) Color.rgb(229, 160, 178) else Color.rgb(157, 66, 101)
            else -> chatColor("ink")
        }
    }

    private fun dialogBuilder(): AlertDialog.Builder {
        val dialogTheme = if (isDarkTheme()) {
            android.R.style.Theme_Material_Dialog_Alert
        } else {
            android.R.style.Theme_Material_Light_Dialog_Alert
        }
        return AlertDialog.Builder(ContextThemeWrapper(this, dialogTheme))
    }

    private fun showThemeSettings() {
        val options = arrayOf("Как в системе", "Светлая", "Тёмная")
        val selected = when (themeMode) {
            "light" -> 1
            "dark" -> 2
            else -> 0
        }
        dialogBuilder()
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

    private fun appendChatMessage(
        role: String,
        value: String,
        code: Boolean = false,
        copyText: String? = null,
        language: String = "text"
    ): LinearLayout? {
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
        val tag = makeText(if (user) "ВЫ" else "SWIN AI", 10f,
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
                setLinkTextColor(chatColor("accent"))
                autoLinkMask = Linkify.WEB_URLS
                linksClickable = true
                movementMethod = LinkMovementMethod.getInstance()
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
            val saveFile = TextView(this).apply {
                text = "＋ Файл"
                textSize = 11f
                setTextColor(chatColor("accent"))
                gravity = Gravity.CENTER
                setPadding(chatDp(8), 0, chatDp(4), 0)
                setOnClickListener { showSaveGeneratedCode(codeText, language) }
            }
            toolbar.addView(saveFile, LinearLayout.LayoutParams(-2, chatDp(28)))
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
                val saveAction = TextView(this).apply {
                    text = "＋  Сохранить в проект"
                    textSize = 12f
                    setTextColor(chatColor("accent"))
                    gravity = Gravity.END or Gravity.CENTER_VERTICAL
                    setPadding(0, chatDp(5), chatDp(2), 0)
                    setOnClickListener { showSaveGeneratedCode(copyText, language) }
                }
                card.addView(saveAction, LinearLayout.LayoutParams(-1, chatDp(30)))
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

        removeWelcomePanel()
        if (!user) {
            val avatar = TextView(this).apply {
                text = "S"
                textSize = 11f
                gravity = Gravity.CENTER
                typeface = Typeface.create("sans-serif", Typeface.BOLD)
                setTextColor(chatColor("sendText"))
                background = chatBg(chatColor("accent"), 10)
            }
            row.addView(avatar, LinearLayout.LayoutParams(chatDp(27), chatDp(27)).apply {
                marginEnd = chatDp(7)
                topMargin = chatDp(4)
            })
        }
        val maxW = (resources.displayMetrics.widthPixels * if (user) 0.88f else 0.80f).toInt()
        row.addView(card, LinearLayout.LayoutParams(maxW, ViewGroup.LayoutParams.WRAP_CONTENT))
        chatMessages.addView(row, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        chatScroll.post { chatScroll.fullScroll(View.FOCUS_DOWN) }
        return row
    }

    private fun showChatHistory() {
        val token = authToken
        if (token.isNullOrBlank()) {
            Toast.makeText(this, "Сначала войди в аккаунт.", Toast.LENGTH_SHORT).show()
            return
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(16), chatDp(8), chatDp(16), chatDp(8))
        }
        val createButton = TextView(this).apply {
            text = "＋  Новый чат"
            textSize = 14f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            gravity = Gravity.CENTER
            setTextColor(chatColor("sendText"))
            background = chatBg(chatColor("accent"), 12)
            setPadding(chatDp(12), 0, chatDp(12), 0)
        }
        content.addView(createButton, LinearLayout.LayoutParams(-1, chatDp(46)))
        val search = makeEdit(
            "Поиск по названию и сообщениям",
            InputType.TYPE_CLASS_TEXT
        ).apply {
            background = chatBg(chatColor("panel"), 12, chatColor("stroke"))
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setPadding(chatDp(12), chatDp(7), chatDp(12), chatDp(7))
            textSize = 14f
        }
        content.addView(search, LinearLayout.LayoutParams(-1, chatDp(48)).apply {
            topMargin = chatDp(10)
        })
        val status = makeText("Загружаю историю…", 12f, chatColor("muted"))
        content.addView(status, LinearLayout.LayoutParams(-1, chatDp(30)).apply {
            topMargin = chatDp(5)
        })
        val list = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        val scroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            addView(list, ViewGroup.LayoutParams(-1, -2))
        }
        content.addView(scroll, LinearLayout.LayoutParams(-1, chatDp(330)))

        val dialog = dialogBuilder()
            .setTitle("История чатов")
            .setView(content)
            .setNegativeButton("Закрыть", null)
            .create()
        createButton.setOnClickListener {
            dialog.dismiss()
            createNewChat()
        }
        dialog.show()

        var sessions = JSONArray()
        fun renderSessions(rawQuery: String) {
            val query = rawQuery.trim().lowercase()
            list.removeAllViews()
            val matches = (0 until sessions.length()).mapNotNull { index ->
                sessions.optJSONObject(index)
            }.filter { item ->
                query.isBlank() ||
                    item.optString("title").lowercase().contains(query) ||
                    item.optString("preview").lowercase().contains(query)
            }
            status.text = when {
                sessions.length() == 0 -> "Сохранённых диалогов пока нет."
                matches.isEmpty() -> "Ничего не найдено."
                else -> "Диалогов: ${matches.size}"
            }
            for (item in matches) {
                val targetId = item.optString("session_id", "")
                if (targetId.isBlank()) continue
                val title = item.optString("title", "Новый чат")
                    .replace(Regex("\\s+"), " ").trim()
                val preview = item.optString("preview", "")
                    .replace(Regex("\\s+"), " ").trim()
                val timestamp = item.optLong("updated_at", 0L)
                val date = if (timestamp > 0L) {
                    java.text.SimpleDateFormat("dd.MM · HH:mm", java.util.Locale.getDefault())
                        .format(java.util.Date(timestamp * 1000L))
                } else {
                    "Недавно"
                }
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(chatDp(13), chatDp(10), chatDp(13), chatDp(10))
                    background = chatBg(
                        chatColor("panel"), 13,
                        if (targetId == sessionId) chatColor("accent") else chatColor("stroke")
                    )
                    isClickable = true
                    isFocusable = true
                    setOnClickListener {
                        dialog.dismiss()
                        openChatSession(targetId)
                    }
                }
                row.addView(makeText(
                    title.ifBlank { "Новый чат" },
                    14f,
                    chatColor("ink")
                ).apply {
                    typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                    maxLines = 1
                })
                row.addView(makeText(
                    preview.ifBlank { "Нет сообщений" },
                    12f,
                    chatColor("muted")
                ).apply {
                    maxLines = 2
                    setPadding(0, chatDp(3), 0, chatDp(5))
                })
                val meta = makeText(
                    "$date  ·  ${item.optInt("message_count", 0)} сообщений",
                    10f,
                    chatColor("accent")
                )
                row.addView(meta)
                list.addView(row, LinearLayout.LayoutParams(-1, -2).apply {
                    bottomMargin = chatDp(8)
                })
            }
        }

        search.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                renderSessions(s?.toString().orEmpty())
            }
            override fun afterTextChanged(s: android.text.Editable?) = Unit
        })

        thread {
            try {
                val response = requestJson(
                    "/memory/sessions?limit=100",
                    JSONObject(),
                    token,
                    method = "GET"
                )
                val loaded = response.optJSONArray("items") ?: JSONArray()
                runOnUiThread {
                    sessions = loaded
                    renderSessions(search.text.toString())
                }
            } catch (error: Exception) {
                runOnUiThread {
                    status.text = "Не удалось загрузить историю: ${error.message ?: "проверь соединение"}"
                }
            }
        }
    }

    private fun openProjectArchivePicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/zip"
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, REQUEST_IMPORT_PROJECT)
        } catch (_: Exception) {
            Toast.makeText(this, "Не удалось открыть выбор ZIP-архива.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun requestProjectArchiveExport(projectId: String) {
        val project = workspaceStore()?.project(projectId) ?: return
        pendingProjectExportId = projectId
        val title = project.optString("name", "swin-project")
            .replace(Regex("[^\\p{L}\\p{N}._-]+"), "_")
        val intent = Intent(Intent.ACTION_CREATE_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "application/zip"
            putExtra(Intent.EXTRA_TITLE, "$title.zip")
        }
        try {
            startActivityForResult(intent, REQUEST_EXPORT_PROJECT)
        } catch (_: Exception) {
            pendingProjectExportId = null
            Toast.makeText(this, "Не удалось открыть сохранение архива.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun exportWorkspaceArchive(projectId: String, target: Uri) {
        try {
            val store = workspaceStore() ?: return
            val project = store.exportProjectData(projectId)
            val manifest = project.toString()
            val files = project.optJSONArray("files") ?: JSONArray()
            val folder = project.optString("project_name", "SWIN-Project")
                .replace(Regex("[^\\p{L}\\p{N}._-]+"), "_")
                .ifBlank { "SWIN-Project" }
            contentResolver.openOutputStream(target)?.use { output ->
                ZipOutputStream(output).use { zip ->
                    zip.putNextEntry(ZipEntry("swin-project.json"))
                    zip.write(manifest.toByteArray(Charsets.UTF_8))
                    zip.closeEntry()
                    for (index in 0 until files.length()) {
                        val file = files.optJSONObject(index) ?: continue
                        val name = file.optString("name")
                        if (name.isBlank() || name == "swin-project.json") continue
                        val entryName = "$folder/$name"
                        zip.putNextEntry(ZipEntry(entryName))
                        zip.write(file.optString("content", "").toByteArray(Charsets.UTF_8))
                        zip.closeEntry()
                    }
                }
            } ?: throw IllegalStateException("Не удалось создать файл архива.")
            Toast.makeText(this, "Проект и отдельные файлы экспортированы в ZIP.", Toast.LENGTH_LONG).show()
        } catch (error: Exception) {
            Toast.makeText(this, "Ошибка экспорта: ${error.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun importWorkspaceArchive(source: Uri) {
        val store = workspaceStore() ?: return
        Toast.makeText(this, "Проверяю ZIP-архив…", Toast.LENGTH_SHORT).show()
        thread {
            try {
                var manifestBytes: ByteArray? = null
                var entries = 0
                var totalBytes = 0
                var looseFileBytes = 0
                val looseFiles = mutableListOf<JSONObject>()
                contentResolver.openInputStream(source)?.use { input ->
                    ZipInputStream(input).use { zip ->
                        val buffer = ByteArray(8192)
                        while (true) {
                            val entry = zip.nextEntry ?: break
                            entries++
                            if (entries > 100) throw IllegalArgumentException("В архиве слишком много файлов.")
                            val name = entry.name
                            if (
                                name.startsWith("/") || name.contains(92.toChar()) ||
                                name.any { it.code == 0 } || name.contains(':') ||
                                name.split("/").any { it == ".." || it == "." }
                            ) {
                                throw IllegalArgumentException("Архив содержит небезопасный путь.")
                            }
                            if (entry.isDirectory) {
                                zip.closeEntry()
                                continue
                            }
                            val isManifest = name == "swin-project.json"
                            if (isManifest && manifestBytes != null) {
                                throw IllegalArgumentException("В архиве несколько манифестов.")
                            }
                            val bytes = ByteArrayOutputStream()
                            while (true) {
                                val count = zip.read(buffer)
                                if (count < 0) break
                                totalBytes += count
                                if (totalBytes > 4_000_000) {
                                    throw IllegalArgumentException("Распакованный архив превышает лимит 4 МБ.")
                                }
                                bytes.write(buffer, 0, count)
                            }
                            val entryBytes = bytes.toByteArray()
                            if (isManifest) {
                                if (entryBytes.size > 1_500_000) {
                                    throw IllegalArgumentException("Манифест проекта слишком большой.")
                                }
                                manifestBytes = entryBytes
                            } else {
                                val filePath = name.trim('/')
                                val pathParts = filePath.split("/")
                                val fileName = pathParts.lastOrNull().orEmpty()
                                val hiddenDirectory = pathParts.dropLast(1).any { it.startsWith(".") }
                                val allowedHiddenFile = fileName == ".gitignore" || fileName == ".env.example"
                                val excludedDirectory = pathParts.any {
                                    it.lowercase() in setOf(
                                        ".git", ".idea", ".gradle", ".venv", ".godot",
                                        "node_modules", "__pycache__", "build", "dist"
                                    ) || it == "__MACOSX"
                                }
                                if (
                                    filePath.isNotBlank() && filePath.length <= 180 &&
                                    !hiddenDirectory && !excludedDirectory &&
                                    (!fileName.startsWith(".") || allowedHiddenFile)
                                ) {
                                    looseFileBytes += entryBytes.size
                                    if (looseFileBytes > 1_500_000 || entryBytes.size > 60_000) {
                                        throw IllegalArgumentException("Файлы архива превышают лимит рабочего пространства.")
                                    }
                                    val decoded = String(entryBytes, Charsets.UTF_8)
                                    if (decoded.contains(0xFFFD.toChar()) || entryBytes.contains(0.toByte())) {
                                        throw IllegalArgumentException("Архив содержит бинарные файлы. Импортируются только текстовые исходники.")
                                    }
                                    if (looseFiles.size >= 80) {
                                        throw IllegalArgumentException("В проекте максимум 80 файлов.")
                                    }
                                    looseFiles.add(
                                        JSONObject()
                                            .put("name", filePath)
                                            .put("language", guessWorkspaceLanguage(filePath))
                                            .put("content", decoded)
                                    )
                                }
                            }
                            zip.closeEntry()
                        }
                    }
                } ?: throw IllegalArgumentException("Не удалось открыть ZIP-архив.")

                val projectName: String
                val sourceFiles: List<JSONObject>
                if (manifestBytes != null) {
                    val manifest = JSONObject(String(manifestBytes!!, Charsets.UTF_8))
                    if (manifest.optString("format") != "swin-ai-project" || manifest.optInt("version") != 1) {
                        throw IllegalArgumentException("Формат архива SWIN не распознан.")
                    }
                    val files = manifest.optJSONArray("files") ?: JSONArray()
                    if (files.length() > 80) throw IllegalArgumentException("В проекте больше 80 файлов.")
                    sourceFiles = (0 until files.length()).mapNotNull { files.optJSONObject(it) }
                    if (sourceFiles.size != files.length()) {
                        throw IllegalArgumentException("Манифест проекта содержит некорректные записи файлов.")
                    }
                    projectName = manifest.optString("project_name", "Импортированный проект")
                } else {
                    if (looseFiles.isEmpty()) {
                        throw IllegalArgumentException("В ZIP нет текстовых файлов проекта.")
                    }
                    sourceFiles = looseFiles
                    projectName = source.lastPathSegment
                        ?.substringAfterLast('/')
                        ?.substringBeforeLast('.')
                        ?.take(64)
                        ?.ifBlank { "Импортированный проект" }
                        ?: "Импортированный проект"
                }
                runOnUiThread {
                    try {
                        val imported = store.importProject(projectName, sourceFiles)
                        Toast.makeText(
                            this,
                            "Импортирован проект: ${imported.optString("name")}",
                            Toast.LENGTH_LONG
                        ).show()
                        showProjectFiles(imported.optString("id"))
                    } catch (error: Exception) {
                        Toast.makeText(this, "Ошибка импорта: ${error.message}", Toast.LENGTH_LONG).show()
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    Toast.makeText(this, "ZIP не импортирован: ${error.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showSaveGeneratedCode(code: String, rawLanguage: String) {
        val store = workspaceStore() ?: return
        if (code.isBlank()) return
        val projects = try {
            store.projects()
        } catch (error: Exception) {
            Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
            return
        }
        val language = rawLanguage.lowercase().ifBlank { "text" }
        val extension = when (language) {
            "python", "py" -> "py"
            "gdscript", "godot" -> "gd"
            "kotlin", "kt" -> "kt"
            "java" -> "java"
            "javascript", "js" -> "js"
            "typescript", "ts" -> "ts"
            "html" -> "html"
            "css" -> "css"
            "json" -> "json"
            "xml" -> "xml"
            "sql" -> "sql"
            "bash", "shell" -> "sh"
            "go" -> "go"
            "rust" -> "rs"
            "c", "cpp", "c++" -> "cpp"
            "csharp", "c#" -> "cs"
            else -> "txt"
        }
        fun chooseFileName(projectId: String) {
            val nameInput = makeEdit("Например, main.$extension", InputType.TYPE_CLASS_TEXT).apply {
                setText("main.$extension")
                background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
                setTextColor(chatColor("ink"))
                setHintTextColor(chatColor("muted"))
                setPadding(chatDp(12), chatDp(8), chatDp(12), chatDp(8))
            }
            val dialog = dialogBuilder().setTitle("Сохранить код в проект")
                .setView(LinearLayout(this).apply {
                    setPadding(chatDp(20), chatDp(8), chatDp(20), 0)
                    addView(nameInput, LinearLayout.LayoutParams(-1, chatDp(50)))
                })
                .setNegativeButton("Отмена", null)
                .setPositiveButton("Сохранить", null)
                .create()
            dialog.setOnShowListener {
                dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                    try {
                        val file = store.createFile(projectId, nameInput.text.toString(), language, code)
                        dialog.dismiss()
                        Toast.makeText(this, "Код сохранён в проект.", Toast.LENGTH_SHORT).show()
                        showWorkspaceEditor(projectId, file)
                    } catch (error: Exception) {
                        nameInput.error = error.message ?: "Не удалось сохранить файл."
                    }
                }
            }
            dialog.show()
        }
        if (projects.isEmpty()) {
            try {
                val project = store.createProject("SWIN Generated")
                chooseFileName(project.optString("id"))
            } catch (error: Exception) {
                Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
            }
            return
        }
        val labels = projects.map { it.optString("name") }.toMutableList()
        labels.add("＋ Создать новый проект")
        dialogBuilder().setTitle("Выбери проект для сохранения")
            .setItems(labels.toTypedArray()) { _, which ->
                if (which < projects.size) {
                    chooseFileName(projects[which].optString("id"))
                } else {
                    val name = "SWIN Generated ${System.currentTimeMillis() / 1000L}"
                    try {
                        val project = store.createProject(name)
                        chooseFileName(project.optString("id"))
                    } catch (error: Exception) {
                        Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
                    }
                }
            }.show()
    }

    private fun showProjectGenerator() {
        if (authToken.isNullOrBlank()) {
            Toast.makeText(this, "Сначала войди в SWIN AI.", Toast.LENGTH_SHORT).show()
            return
        }
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(20), chatDp(8), chatDp(20), chatDp(2))
        }
        val taskInput = EditText(this).apply {
            hint = "Опиши приложение, игру, сайт или инструмент. Укажи функции и платформу…"
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            textSize = 14f
            minLines = 5
            maxLines = 9
            gravity = Gravity.TOP or Gravity.START
            inputType = InputType.TYPE_CLASS_TEXT or
                InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                InputType.TYPE_TEXT_FLAG_CAP_SENTENCES
            setPadding(chatDp(12), chatDp(10), chatDp(12), chatDp(10))
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
        }
        panel.addView(taskInput, LinearLayout.LayoutParams(-1, chatDp(150)))
        panel.addView(makeText(
            "SWIN создаст локальный стартовый шаблон Godot 4, Python или Web (до 12 файлов). Это основа проекта, а не готовая реализация всех функций. Никаких внешних AI API.",
            11f,
            chatColor("muted")
        ).apply {
            setPadding(0, chatDp(8), 0, chatDp(5))
        })
        val options = listOf("Автоопределение" to "auto") + workspaceLanguages()
        val spinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                options.map { it.first }
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(0)
            background = chatBg(chatColor("panel"), 10, chatColor("stroke"))
        }
        panel.addView(spinner, LinearLayout.LayoutParams(-1, chatDp(48)))
        val dialog = dialogBuilder()
            .setTitle("Локальный генератор проекта")
            .setView(panel)
            .setNegativeButton("Отмена", null)
            .setPositiveButton("Создать проект", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val task = taskInput.text.toString().trim()
                if (task.length < 10) {
                    taskInput.error = "Опиши проект подробнее."
                    return@setOnClickListener
                }
                val token = authToken
                if (token.isNullOrBlank()) {
                    taskInput.error = "Сессия истекла. Войди снова."
                    return@setOnClickListener
                }
                val language = options.getOrNull(spinner.selectedItemPosition)?.second ?: "auto"
                val payload = JSONObject()
                    .put("task", task)
                    .put("language", language)
                    .put("response_language", responseLanguage)
                    .put("project_context", if (::projectContextInput.isInitialized) projectContextInput.text.toString().take(8000) else "")
                    .put("session_id", sessionId)
                    .put("max_files", 12)
                dialog.dismiss()
                val progress = ProgressDialog(this).apply {
                    setMessage("SWIN создаёт локальный шаблон и проверяет исходники…")
                    setCancelable(false)
                    show()
                }
                thread {
                    try {
                        val response = requestJson("/code/project", payload, token)
                        val source = response.optJSONArray("files")
                            ?: throw IllegalStateException("Сервер не вернул файлы проекта.")
                        if (source.length() !in 1..20) {
                            throw IllegalStateException("Сервер вернул недопустимое количество файлов.")
                        }
                        val importFiles = ArrayList<JSONObject>()
                        for (index in 0 until source.length()) {
                            val file = source.optJSONObject(index)
                                ?: throw IllegalStateException("Повреждена запись файла №${index + 1}.")
                            val name = file.optString("path")
                            val content = file.optString("content")
                            if (name.isBlank() || content.isBlank()) {
                                throw IllegalStateException("Файл №${index + 1} не содержит имени или исходника.")
                            }
                            importFiles.add(
                                JSONObject()
                                    .put("name", name)
                                    .put("language", file.optString("language", guessWorkspaceLanguage(name)))
                                    .put("content", content)
                            )
                        }
                        val projectName = response.optString("project_name", "SWIN Project")
                        runOnUiThread {
                            try {
                                val store = workspaceStore()
                                    ?: throw IllegalStateException("Локальное рабочее пространство недоступно.")
                                val saved = store.importProject(projectName, importFiles)
                                progress.dismiss()
                                val validation = response.optJSONObject("validation")
                                val warning = validation?.optString("message").orEmpty()
                                val scope = response.optString("scope", "template_only")
                                val summary = response.optString("summary", "")
                                Toast.makeText(
                                    this,
                                    "Создан стартовый проект «${saved.optString("name")}»: ${importFiles.size} файлов. $scope. $warning",
                                    Toast.LENGTH_LONG
                                ).show()
                                if (summary.isNotBlank()) {
                                    appendChatMessage("assistant", summary)
                                }
                                showProjectFiles(saved.optString("id"))
                            } catch (error: Exception) {
                                progress.dismiss()
                                Toast.makeText(this, "Не удалось сохранить проект: ${error.message}", Toast.LENGTH_LONG).show()
                            }
                        }
                    } catch (error: Exception) {
                        runOnUiThread {
                            progress.dismiss()
                            Toast.makeText(
                                this,
                                "Генерация не завершена: ${error.message}",
                                Toast.LENGTH_LONG
                            ).show()
                        }
                    }
                }
            }
        }
        dialog.show()
    }

    private fun workspaceStore(): WorkspaceStore? {
        if (userEmail.isBlank()) {
            Toast.makeText(this, "Сначала войди в аккаунт.", Toast.LENGTH_SHORT).show()
            return null
        }
        return try {
            WorkspaceStore(this, userEmail)
        } catch (error: Exception) {
            Toast.makeText(this, error.message ?: "Хранилище недоступно.", Toast.LENGTH_LONG).show()
            null
        }
    }

    private fun showWorkspace() {
        val store = workspaceStore() ?: return
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(16), chatDp(8), chatDp(16), chatDp(8))
        }
        root.addView(makeText(
            "Локальные проекты и файлы. Исходники шифруются ключом Android Keystore и доступны только этому аккаунту на этом телефоне.",
            12f,
            chatColor("muted")
        ).apply { setLineSpacing(chatDp(2).toFloat(), 1f) })
        val create = TextView(this).apply {
            text = "＋  Создать проект"
            gravity = Gravity.CENTER
            textSize = 14f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(chatColor("sendText"))
            background = chatBg(chatColor("accent"), 12)
        }
        val workspaceActions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        workspaceActions.addView(create, LinearLayout.LayoutParams(0, chatDp(46), 1f))
        val importButton = TextView(this).apply {
            text = "Импорт ZIP"
            gravity = Gravity.CENTER
            textSize = 13f
            setTextColor(chatColor("accent"))
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
        }
        workspaceActions.addView(importButton, LinearLayout.LayoutParams(chatDp(112), chatDp(46)).apply {
            marginStart = chatDp(8)
        })
        root.addView(workspaceActions, LinearLayout.LayoutParams(-1, chatDp(46)).apply {
            topMargin = chatDp(12)
        })
        val generateProjectButton = TextView(this).apply {
            text = "✦  Создать полный проект с SWIN"
            gravity = Gravity.CENTER
            textSize = 13f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(chatColor("accent"))
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
            setOnClickListener { showProjectGenerator() }
        }
        root.addView(generateProjectButton, LinearLayout.LayoutParams(-1, chatDp(46)).apply {
            topMargin = chatDp(8)
        })
        val search = makeEdit("Найти проект…", InputType.TYPE_CLASS_TEXT).apply {
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setPadding(chatDp(12), chatDp(6), chatDp(12), chatDp(6))
            textSize = 14f
        }
        root.addView(search, LinearLayout.LayoutParams(-1, chatDp(46)).apply {
            topMargin = chatDp(8)
        })
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val scroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            addView(list, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(scroll, LinearLayout.LayoutParams(-1, chatDp(340)).apply {
            topMargin = chatDp(8)
        })
        val dialog = dialogBuilder().setTitle("Рабочее пространство")
            .setView(root).setNegativeButton("Закрыть", null).create()
        dialog.show()
        create.setOnClickListener {
            dialog.dismiss()
            showCreateWorkspaceProject()
        }
        importButton.setOnClickListener {
            dialog.dismiss()
            openProjectArchivePicker()
        }

        fun renderProjects(queryText: String) {
            list.removeAllViews()
            val query = queryText.trim().lowercase()
            val projects = try {
                store.projects().filter { it.optString("name").lowercase().contains(query) }
            } catch (error: Exception) {
                Toast.makeText(this, error.message ?: "Ошибка хранилища.", Toast.LENGTH_LONG).show()
                emptyList()
            }
            if (projects.isEmpty()) {
                list.addView(makeText(
                    if (query.isBlank()) "Пока нет проектов. Создай первый проект и добавь файлы."
                    else "Проекты не найдены.",
                    13f,
                    chatColor("muted")
                ).apply {
                    gravity = Gravity.CENTER
                    setPadding(chatDp(14), chatDp(28), chatDp(14), chatDp(28))
                })
                return
            }
            for (project in projects) {
                val projectId = project.optString("id")
                val files = project.optJSONArray("files")?.length() ?: 0
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(chatDp(14), chatDp(12), chatDp(14), chatDp(12))
                    background = chatBg(chatColor("panel"), 13, chatColor("stroke"))
                    isClickable = true
                    isFocusable = true
                    setOnClickListener {
                        dialog.dismiss()
                        showProjectFiles(projectId)
                    }
                    setOnLongClickListener {
                        dialogBuilder().setTitle("Удалить проект?")
                            .setMessage("Проект «${project.optString("name")}» и все его файлы будут удалены с этого телефона.")
                            .setNegativeButton("Отмена", null)
                            .setPositiveButton("Удалить") { _, _ ->
                                try {
                                    store.deleteProject(projectId)
                                    renderProjects(search.text.toString())
                                } catch (error: Exception) {
                                    Toast.makeText(this@MainActivity, error.message, Toast.LENGTH_LONG).show()
                                }
                            }.show()
                        true
                    }
                }
                row.addView(makeText(project.optString("name"), 15f, chatColor("ink")).apply {
                    typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                })
                row.addView(makeText(
                    "$files файлов  ·  удержи, чтобы удалить",
                    11f,
                    chatColor("muted")
                ).apply { setPadding(0, chatDp(4), 0, 0) })
                list.addView(row, LinearLayout.LayoutParams(-1, -2).apply {
                    bottomMargin = chatDp(8)
                })
            }
        }

        search.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                renderProjects(s?.toString().orEmpty())
            }
            override fun afterTextChanged(s: android.text.Editable?) = Unit
        })
        renderProjects("")
    }

    private fun showCreateWorkspaceProject() {
        val input = makeEdit("Например, MyGame", InputType.TYPE_CLASS_TEXT).apply {
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setPadding(chatDp(12), chatDp(8), chatDp(12), chatDp(8))
        }
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(20), chatDp(8), chatDp(20), 0)
            addView(input, LinearLayout.LayoutParams(-1, chatDp(50)))
        }
        val dialog = dialogBuilder().setTitle("Новый проект")
            .setView(box)
            .setNegativeButton("Отмена", null)
            .setPositiveButton("Создать", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    val project = workspaceStore()?.createProject(input.text.toString()) ?: return@setOnClickListener
                    dialog.dismiss()
                    showProjectFiles(project.optString("id"))
                } catch (error: Exception) {
                    input.error = error.message ?: "Не удалось создать проект."
                }
            }
        }
        dialog.show()
    }

    private fun showProjectFiles(projectId: String) {
        val store = workspaceStore() ?: return
        val project = try {
            store.project(projectId)
        } catch (error: Exception) {
            Toast.makeText(this, error.message, Toast.LENGTH_LONG).show()
            null
        } ?: run {
            Toast.makeText(this, "Проект не найден.", Toast.LENGTH_SHORT).show()
            return
        }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(16), chatDp(8), chatDp(16), chatDp(8))
        }
        root.addView(makeText(
            "Файлы проекта «${project.optString("name")}»",
            13f,
            chatColor("muted")
        ))
        val create = TextView(this).apply {
            text = "＋  Новый файл"
            gravity = Gravity.CENTER
            textSize = 14f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(chatColor("sendText"))
            background = chatBg(chatColor("accent"), 12)
        }
        val fileActions = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        fileActions.addView(create, LinearLayout.LayoutParams(0, chatDp(44), 1f))
        val exportButton = TextView(this).apply {
            text = "Экспорт ZIP"
            gravity = Gravity.CENTER
            textSize = 12f
            setTextColor(chatColor("accent"))
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
            setOnClickListener { requestProjectArchiveExport(projectId) }
        }
        fileActions.addView(exportButton, LinearLayout.LayoutParams(chatDp(112), chatDp(44)).apply {
            marginStart = chatDp(8)
        })
        root.addView(fileActions, LinearLayout.LayoutParams(-1, chatDp(44)).apply {
            topMargin = chatDp(10)
        })
        val list = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        val scroll = ScrollView(this).apply {
            isVerticalScrollBarEnabled = false
            addView(list, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(scroll, LinearLayout.LayoutParams(-1, chatDp(360)).apply {
            topMargin = chatDp(8)
        })
        val dialog = dialogBuilder().setTitle("Файлы проекта")
            .setView(root).setNegativeButton("Назад", null).create()
        dialog.show()

        val files = project.optJSONArray("files") ?: JSONArray()
        if (files.length() == 0) {
            list.addView(makeText("Файлов пока нет. Создай первый файл, например main.gd или main.py.", 13f, chatColor("muted")).apply {
                gravity = Gravity.CENTER
                setPadding(chatDp(14), chatDp(28), chatDp(14), chatDp(28))
            })
        } else {
            val sorted = (0 until files.length()).mapNotNull { files.optJSONObject(it) }
                .sortedByDescending { it.optLong("updated_at", 0L) }
            for (file in sorted) {
                val fileId = file.optString("id")
                val name = file.optString("name")
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(chatDp(13), chatDp(11), chatDp(13), chatDp(11))
                    background = chatBg(chatColor("panel"), 12, chatColor("stroke"))
                    isClickable = true
                    isFocusable = true
                    setOnClickListener {
                        dialog.dismiss()
                        showWorkspaceEditor(projectId, file)
                    }
                    setOnLongClickListener {
                        dialogBuilder().setTitle("Удалить файл?")
                            .setMessage("Файл «$name» будет удалён из проекта.")
                            .setNegativeButton("Отмена", null)
                            .setPositiveButton("Удалить") { _, _ ->
                                try {
                                    store.deleteFile(projectId, fileId)
                                    dialog.dismiss()
                                    showProjectFiles(projectId)
                                } catch (error: Exception) {
                                    Toast.makeText(this@MainActivity, error.message, Toast.LENGTH_LONG).show()
                                }
                            }.show()
                        true
                    }
                }
                row.addView(makeText(name, 14f, chatColor("ink")).apply {
                    typeface = Typeface.create("monospace", Typeface.BOLD)
                })
                row.addView(makeText(
                    file.optString("language", "text").uppercase() +
                        "  ·  ${file.optString("content").length} символов  ·  удержи, чтобы удалить",
                    10f,
                    chatColor("muted")
                ).apply { setPadding(0, chatDp(4), 0, 0) })
                list.addView(row, LinearLayout.LayoutParams(-1, -2).apply {
                    bottomMargin = chatDp(7)
                })
            }
        }
        create.setOnClickListener {
            dialog.dismiss()
            showCreateWorkspaceFile(projectId)
        }
    }

    private fun workspaceLanguages(): List<Pair<String, String>> = listOf(
        "Godot 4 / GDScript" to "gdscript",
        "Python" to "python",
        "Kotlin" to "kotlin",
        "Java" to "java",
        "JavaScript" to "javascript",
        "TypeScript" to "typescript",
        "JSON" to "json",
        "HTML" to "html",
        "CSS" to "css",
        "XML" to "xml",
        "SQL" to "sql",
        "C / C++" to "cpp",
        "C#" to "csharp",
        "Go" to "go",
        "Rust" to "rust",
        "Shell" to "bash",
        "Текст" to "text"
    )

    private fun guessWorkspaceLanguage(fileName: String): String {
        return when (fileName.substringAfterLast('.', "").lowercase()) {
            "gd" -> "gdscript"
            "py" -> "python"
            "kt", "kts" -> "kotlin"
            "java" -> "java"
            "js", "mjs", "cjs" -> "javascript"
            "ts", "tsx" -> "typescript"
            "json" -> "json"
            "html", "htm" -> "html"
            "css", "scss" -> "css"
            "xml", "svg" -> "xml"
            "sql" -> "sql"
            "c", "h", "cpp", "cc", "hpp" -> "cpp"
            "cs" -> "csharp"
            "go" -> "go"
            "rs" -> "rust"
            "sh", "bash" -> "bash"
            else -> "text"
        }
    }

    private fun showCreateWorkspaceFile(projectId: String) {
        val store = workspaceStore() ?: return
        val box = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(20), chatDp(8), chatDp(20), 0)
        }
        val nameInput = makeEdit("Например, main.gd", InputType.TYPE_CLASS_TEXT).apply {
            background = chatBg(chatColor("panel"), 11, chatColor("stroke"))
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setPadding(chatDp(12), chatDp(8), chatDp(12), chatDp(8))
        }
        box.addView(nameInput, LinearLayout.LayoutParams(-1, chatDp(50)))
        val options = workspaceLanguages()
        val languageSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                options.map { it.first }
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(0)
            background = chatBg(chatColor("panel"), 10, chatColor("stroke"))
        }
        box.addView(languageSpinner, LinearLayout.LayoutParams(-1, chatDp(48)).apply {
            topMargin = chatDp(10)
        })
        nameInput.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                val guessed = guessWorkspaceLanguage(s?.toString().orEmpty())
                val index = options.indexOfFirst { it.second == guessed }
                if (index >= 0) languageSpinner.setSelection(index)
            }
            override fun afterTextChanged(s: android.text.Editable?) = Unit
        })
        val dialog = dialogBuilder().setTitle("Новый файл")
            .setView(box)
            .setNegativeButton("Отмена", null)
            .setPositiveButton("Создать", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    val language = options.getOrNull(languageSpinner.selectedItemPosition)?.second ?: "text"
                    val file = store.createFile(projectId, nameInput.text.toString(), language)
                    dialog.dismiss()
                    showWorkspaceEditor(projectId, file)
                } catch (error: Exception) {
                    nameInput.error = error.message ?: "Не удалось создать файл."
                }
            }
        }
        dialog.show()
    }

    private fun showWorkspaceEditor(projectId: String, file: JSONObject) {
        val store = workspaceStore() ?: return
        val fileId = file.optString("id")
        var dirty = false
        var currentName = file.optString("name")
        var currentLanguage = file.optString("language", "text")
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(12), chatDp(8), chatDp(12), chatDp(8))
        }
        val top = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val filenameInput = makeEdit("Имя файла", InputType.TYPE_CLASS_TEXT).apply {
            setText(currentName)
            background = chatBg(chatColor("panel"), 10, chatColor("stroke"))
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setPadding(chatDp(10), chatDp(4), chatDp(10), chatDp(4))
            textSize = 13f
        }
        top.addView(filenameInput, LinearLayout.LayoutParams(0, chatDp(46), 1f))
        val options = workspaceLanguages()
        val languageSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                options.map { it.first }
            ).also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            setSelection(options.indexOfFirst { it.second == currentLanguage }.coerceAtLeast(0))
            background = chatBg(chatColor("panel"), 10, chatColor("stroke"))
        }
        top.addView(languageSpinner, LinearLayout.LayoutParams(chatDp(142), chatDp(46)).apply {
            marginStart = chatDp(7)
        })
        root.addView(top)

        val editorRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            background = chatBg(chatColor("codeBubble"), 11, chatColor("stroke"))
        }
        val lineNumbers = makeText("1", 12f, chatColor("muted")).apply {
            typeface = Typeface.create("monospace", Typeface.NORMAL)
            gravity = Gravity.TOP or Gravity.END
            setPadding(chatDp(7), chatDp(12), chatDp(8), chatDp(12))
            setLineSpacing(chatDp(3).toFloat(), 1f)
        }
        editorRow.addView(lineNumbers, LinearLayout.LayoutParams(chatDp(38), -1))
        val editor = EditText(this).apply {
            setText(file.optString("content", ""))
            textSize = 13f
            typeface = Typeface.create("monospace", Typeface.NORMAL)
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            hint = "// Начни писать код…"
            gravity = Gravity.TOP or Gravity.START
            inputType = InputType.TYPE_CLASS_TEXT or
                InputType.TYPE_TEXT_FLAG_MULTI_LINE or
                InputType.TYPE_TEXT_FLAG_NO_SUGGESTIONS
            setHorizontallyScrolling(true)
            isVerticalScrollBarEnabled = false
            setPadding(chatDp(9), chatDp(12), chatDp(12), chatDp(12))
            background = ColorDrawable(Color.TRANSPARENT)
            minLines = 18
            maxLines = 600
            setLineSpacing(chatDp(3).toFloat(), 1f)
        }
        editorRow.addView(editor, LinearLayout.LayoutParams(0, -2, 1f))
        val codeScroll = ScrollView(this).apply {
            isFillViewport = false
            isVerticalScrollBarEnabled = true
            addView(editorRow, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(codeScroll, LinearLayout.LayoutParams(-1, 0, 1f).apply {
            topMargin = chatDp(9)
        })
        val status = makeText("", 11f, chatColor("muted")).apply {
            setPadding(chatDp(3), chatDp(6), chatDp(3), chatDp(2))
        }
        root.addView(status)

        fun refreshEditorDecorations(text: CharSequence) {
            val lines = text.count { it == '\n' } + 1
            lineNumbers.text = (1..lines).joinToString("\n")
            status.text = "$lines строк  ·  ${text.length} символов" +
                if (dirty) "  ·  не сохранено" else "  ·  сохранено"
        }
        fun selectedLanguage(): String =
            options.getOrNull(languageSpinner.selectedItemPosition)?.second ?: currentLanguage
        fun saveCurrentFile(): Boolean {
            return try {
                val newName = filenameInput.text.toString()
                val newLanguage = selectedLanguage()
                val newContent = editor.text.toString()
                store.saveFile(projectId, fileId, newName, newLanguage, newContent)
                currentName = newName
                currentLanguage = newLanguage
                file.put("name", newName).put("language", newLanguage)
                    .put("content", newContent)
                dirty = false
                refreshEditorDecorations(editor.text)
                true
            } catch (error: Exception) {
                filenameInput.error = error.message ?: "Не удалось сохранить файл."
                false
            }
        }
        editor.addTextChangedListener(object : android.text.TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) = Unit
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                dirty = true
            }
            override fun afterTextChanged(s: android.text.Editable?) {
                if (s != null) {
                    CodeSyntaxHighlighter.apply(s, selectedLanguage(), ::chatColor)
                    refreshEditorDecorations(s)
                }
            }
        })
        languageSpinner.onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
            override fun onNothingSelected(parent: android.widget.AdapterView<*>?) = Unit
            override fun onItemSelected(
                parent: android.widget.AdapterView<*>?,
                view: View?,
                position: Int,
                id: Long
            ) {
                editor.text?.let { CodeSyntaxHighlighter.apply(it, selectedLanguage(), ::chatColor) }
                if (currentLanguage != selectedLanguage()) dirty = true
                refreshEditorDecorations(editor.text)
            }
        }
        refreshEditorDecorations(editor.text)
        CodeSyntaxHighlighter.apply(editor.text, currentLanguage, ::chatColor)

        val dialog = dialogBuilder()
            .setTitle("Редактор кода")
            .setView(root)
            .setNegativeButton("Закрыть", null)
            .setPositiveButton("Сохранить", null)
            .setNeutralButton("Передать SWIN", null)
            .create()
        dialog.setOnShowListener {
            dialog.window?.setLayout(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                if (saveCurrentFile()) {
                    Toast.makeText(this, "Файл сохранён", Toast.LENGTH_SHORT).show()
                }
            }
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL).setOnClickListener {
                if (!saveCurrentFile()) return@setOnClickListener
                val source = editor.text.toString()
                val contextText = "Проект: ${store.project(projectId)?.optString("name") ?: ""}\n" +
                    "Файл: $currentName\nЯзык: $currentLanguage\n\n" + source
                if (::projectContextInput.isInitialized) {
                    projectContextInput.setText(contextText.take(4000))
                    if (contextText.length > 4000) {
                        Toast.makeText(
                            this,
                            "Файл сохранён полностью; в контекст SWIN помещены первые 4000 символов.",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
                isCodeMode = true
                updateCodingControls()
                codeModeHeader?.text = "Код ✓"
                dialog.dismiss()
                focusComposer("Проверь файл $currentName. Найди ошибки, объясни их и предложи исправления.", coding = true)
            }
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                if (dirty) {
                    dialogBuilder().setTitle("Есть несохранённые изменения")
                        .setMessage("Сохранить изменения перед закрытием редактора?")
                        .setNegativeButton("Без сохранения") { _, _ -> dialog.dismiss() }
                        .setPositiveButton("Сохранить") { _, _ ->
                            if (saveCurrentFile()) dialog.dismiss()
                        }.show()
                } else {
                    dialog.dismiss()
                }
            }
        }
        dialog.show()
    }

    private fun showChatSettings() {
        val options = arrayOf(
            "История чатов",
            "Новый чат",
            "Рабочее пространство",
            "Память SWIN",
            "Язык ответа",
            "Оформление",
            "Адрес сервера",
            "Очистить историю чата",
            "Выйти из аккаунта"
        )
        dialogBuilder()
            .setTitle("Настройки")
            .setItems(options) { _, which ->
                when (which) {
                    0 -> showChatHistory()
                    1 -> createNewChat()
                    2 -> showWorkspace()
                    3 -> showMemorySettings()
                    4 -> showResponseLanguageSettings()
                    5 -> showThemeSettings()
                    6 -> showEndpointSettings()
                    7 -> confirmClearChatMemory()
                    8 -> logout()
                }
            }
            .setNegativeButton("Закрыть", null)
            .show()
    }


    private fun codingUiLabels(language: String): Map<String, String> {
        val code = language.lowercase().substringBefore("-")
        return when (code) {
            "en" -> mapOf(
                "heading" to "SWIN AI • CODING MODE", "status" to "Status",
                "syntax" to "Syntax check", "recheck" to "VERIFICATION",
                "research" to "Web research", "static" to "Static analysis",
                "cross" to "Reference cross-check", "runtime" to "Runtime tests",
                "code" to "CODE",
                "noCode" to "No code passed the available checks. Clarify the task and try again.",
                "plan" to "Work plan", "stages" to "Stages", "sources" to "Documentation sources",
                "webSource" to "Web source", "time" to "Processing time", "ms" to "ms"
            )
            "de" -> mapOf(
                "heading" to "SWIN AI • ENTWICKLUNGSMODUS", "status" to "Status",
                "syntax" to "Syntaxprüfung", "recheck" to "ÜBERPRÜFUNG",
                "research" to "Webrecherche", "static" to "Statische Analyse",
                "cross" to "Quellenabgleich", "runtime" to "Laufzeittests",
                "code" to "CODE",
                "noCode" to "Es wurde kein Code erstellt, der die verfügbaren Prüfungen bestanden hat. Präzisiere die Aufgabe und versuche es erneut.",
                "plan" to "Arbeitsplan", "stages" to "Phasen", "sources" to "Dokumentationsquellen",
                "webSource" to "Webquelle", "time" to "Verarbeitungszeit", "ms" to "ms"
            )
            "es" -> mapOf(
                "heading" to "SWIN AI • MODO DE DESARROLLO", "status" to "Estado",
                "syntax" to "Comprobación de sintaxis", "recheck" to "VERIFICACIÓN",
                "research" to "Búsqueda web", "static" to "Análisis estático",
                "cross" to "Contraste de fuentes", "runtime" to "Pruebas de ejecución",
                "code" to "CÓDIGO",
                "noCode" to "No se pudo preparar código que superara las comprobaciones disponibles. Aclara la tarea e inténtalo de nuevo.",
                "plan" to "Plan de trabajo", "stages" to "Etapas", "sources" to "Fuentes de documentación",
                "webSource" to "Fuente web", "time" to "Tiempo de procesamiento", "ms" to "ms"
            )
            "fr" -> mapOf(
                "heading" to "SWIN AI • MODE DÉVELOPPEMENT", "status" to "Statut",
                "syntax" to "Vérification syntaxique", "recheck" to "VÉRIFICATION",
                "research" to "Recherche Web", "static" to "Analyse statique",
                "cross" to "Vérification des sources", "runtime" to "Tests d’exécution",
                "code" to "CODE",
                "noCode" to "Aucun code n’a réussi les vérifications disponibles. Précisez la tâche et réessayez.",
                "plan" to "Plan de travail", "stages" to "Étapes", "sources" to "Sources documentaires",
                "webSource" to "Source Web", "time" to "Temps de traitement", "ms" to "ms"
            )
            "zh" -> mapOf(
                "heading" to "SWIN AI • 开发模式", "status" to "状态",
                "syntax" to "语法检查", "recheck" to "验证",
                "research" to "网页检索", "static" to "静态分析",
                "cross" to "参考资料核对", "runtime" to "运行测试",
                "code" to "代码",
                "noCode" to "没有代码通过当前可用的检查。请补充任务要求后重试。",
                "plan" to "工作计划", "stages" to "处理阶段", "sources" to "文档来源",
                "webSource" to "网页来源", "time" to "处理时间", "ms" to "毫秒"
            )
            "uk" -> mapOf(
                "heading" to "SWIN AI • РЕЖИМ РОЗРОБКИ", "status" to "Статус",
                "syntax" to "Перевірка синтаксису", "recheck" to "ПЕРЕВІРКА",
                "research" to "Пошук в інтернеті", "static" to "Статичний аналіз",
                "cross" to "Зіставлення джерел", "runtime" to "Запуск тестів",
                "code" to "КОД",
                "noCode" to "Не вдалося підготувати код, що пройшов доступні перевірки. Уточни завдання та спробуй ще раз.",
                "plan" to "План роботи", "stages" to "Етапи", "sources" to "Джерела документації",
                "webSource" to "Вебджерело", "time" to "Час обробки", "ms" to "мс"
            )
            "pt" -> mapOf(
                "heading" to "SWIN AI • MODO DE DESENVOLVIMENTO", "status" to "Estado",
                "syntax" to "Verificação de sintaxe", "recheck" to "VERIFICAÇÃO",
                "research" to "Pesquisa na web", "static" to "Análise estática",
                "cross" to "Conferência de fontes", "runtime" to "Testes de execução",
                "code" to "CÓDIGO",
                "noCode" to "Nenhum código passou pelas verificações disponíveis. Esclareça a tarefa e tente novamente.",
                "plan" to "Plano de trabalho", "stages" to "Etapas", "sources" to "Fontes da documentação",
                "webSource" to "Fonte da web", "time" to "Tempo de processamento", "ms" to "ms"
            )
            "it" -> mapOf(
                "heading" to "SWIN AI • MODALITÀ SVILUPPO", "status" to "Stato",
                "syntax" to "Controllo sintattico", "recheck" to "VERIFICA",
                "research" to "Ricerca sul web", "static" to "Analisi statica",
                "cross" to "Confronto delle fonti", "runtime" to "Test di esecuzione",
                "code" to "CODICE",
                "noCode" to "Nessun codice ha superato i controlli disponibili. Chiarisci il compito e riprova.",
                "plan" to "Piano di lavoro", "stages" to "Fasi", "sources" to "Fonti della documentazione",
                "webSource" to "Fonte web", "time" to "Tempo di elaborazione", "ms" to "ms"
            )
            "ja" -> mapOf(
                "heading" to "SWIN AI • 開発モード", "status" to "状態",
                "syntax" to "構文チェック", "recheck" to "検証",
                "research" to "Web 検索", "static" to "静的解析",
                "cross" to "参考資料の照合", "runtime" to "実行テスト",
                "code" to "コード",
                "noCode" to "利用可能なチェックに合格したコードを作成できませんでした。タスクを詳しくして再試行してください。",
                "plan" to "作業計画", "stages" to "処理段階", "sources" to "ドキュメントの参照元",
                "webSource" to "Web ソース", "time" to "処理時間", "ms" to "ミリ秒"
            )
            else -> mapOf(
                "heading" to "SWIN AI • РЕЖИМ РАЗРАБОТКИ", "status" to "Статус",
                "syntax" to "Проверка синтаксиса", "recheck" to "ПЕРЕПРОВЕРКА",
                "research" to "Поиск в интернете", "static" to "Статический анализ",
                "cross" to "Сверка источников", "runtime" to "Запуск тестов",
                "code" to "КОД",
                "noCode" to "Не удалось подготовить код, который прошёл проверку. Уточни задачу или попробуй снова.",
                "plan" to "План работы", "stages" to "Этапы", "sources" to "Источники документации",
                "webSource" to "Веб-источник", "time" to "Время обработки", "ms" to "мс"
            )
        }
    }

    private fun showMemorySettings() {
        val token = authToken
        if (token.isNullOrBlank()) {
            Toast.makeText(this, "Сначала войди в аккаунт.", Toast.LENGTH_SHORT).show()
            return
        }
        thread {
            try {
                val response = requestJson("/memory/facts", JSONObject(), token, method = "GET")
                val items = response.optJSONArray("items") ?: JSONArray()
                runOnUiThread { renderMemorySettings(items) }
            } catch (error: Exception) {
                runOnUiThread {
                    dialogBuilder()
                        .setTitle("Память SWIN")
                        .setMessage("Не удалось загрузить сохранённые факты.\n" + (error.message ?: "Проверь соединение с сервером."))
                        .setPositiveButton("Повторить") { _, _ -> showMemorySettings() }
                        .setNegativeButton("Закрыть", null)
                        .show()
                }
            }
        }
    }

    private fun renderMemorySettings(items: JSONArray) {
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(18), chatDp(10), chatDp(18), chatDp(8))
        }
        root.addView(makeText(
            "Факты сохраняются в твоём аккаунте и помогают SWIN учитывать предпочтения и контекст проектов.",
            13f,
            chatColor("muted")
        ))
        val list = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        if (items.length() == 0) {
            val empty = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setPadding(chatDp(16), chatDp(22), chatDp(16), chatDp(22))
                background = chatBg(chatColor("panel"), 15, chatColor("stroke"))
            }
            empty.addView(makeText("Пока ничего не сохранено", 15f, chatColor("ink")).apply {
                gravity = Gravity.CENTER
                typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            })
            empty.addView(makeText("Например: «Запомни, что мой проект использует Godot 4».", 12f, chatColor("muted")).apply {
                gravity = Gravity.CENTER
                setPadding(0, chatDp(6), 0, 0)
            })
            list.addView(empty)
        } else {
            for (index in 0 until items.length()) {
                val item = items.optJSONObject(index) ?: continue
                val key = item.optString("memory_key", "")
                val value = item.optString("value", "")
                val category = when (item.optString("category", "note")) {
                    "preference" -> "ПРЕДПОЧТЕНИЕ"
                    "project" -> "ПРОЕКТ"
                    else -> "ЗАМЕТКА"
                }
                val card = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    setPadding(chatDp(13), chatDp(10), chatDp(13), chatDp(8))
                    background = chatBg(chatColor("panel"), 14, chatColor("stroke"))
                }
                val top = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = Gravity.CENTER_VERTICAL
                }
                top.addView(makeText(category, 10f, chatColor("accent")).apply {
                    letterSpacing = 0.08f
                    typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
                }, LinearLayout.LayoutParams(0, chatDp(24), 1f))
                val remove = TextView(this).apply {
                    text = "Удалить"
                    textSize = 12f
                    setTextColor(chatColor("danger"))
                    gravity = Gravity.CENTER
                    setPadding(chatDp(8), 0, 0, 0)
                    setOnClickListener { deleteMemoryFact(key) }
                }
                top.addView(remove, LinearLayout.LayoutParams(-2, chatDp(30)))
                card.addView(top)
                card.addView(makeText(value, 14f, chatColor("ink")).apply {
                    setTextIsSelectable(true)
                    setPadding(0, chatDp(3), 0, chatDp(2))
                })
                list.addView(card, LinearLayout.LayoutParams(-1, -2).apply {
                    bottomMargin = chatDp(8)
                })
            }
        }
        val scroll = ScrollView(this).apply {
            isFillViewport = false
            isVerticalScrollBarEnabled = false
            addView(list, ViewGroup.LayoutParams(-1, -2))
        }
        root.addView(scroll, LinearLayout.LayoutParams(-1, chatDp(300)).apply {
            topMargin = chatDp(12)
        })
        val dialog = dialogBuilder()
            .setTitle("Долговременная память")
            .setView(root)
            .setPositiveButton("Готово", null)
            .setNeutralButton("＋ Добавить факт", null)
            .create()
        memoryDialog = dialog
        dialog.setOnDismissListener {
            if (memoryDialog === dialog) memoryDialog = null
        }
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL)?.setTextColor(chatColor("accent"))
            dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setTextColor(chatColor("accent"))
            dialog.getButton(AlertDialog.BUTTON_NEUTRAL)?.setOnClickListener {
                dialog.dismiss()
                showAddMemoryDialog()
            }
        }
        dialog.show()
    }

    private fun showAddMemoryDialog() {
        val input = EditText(this).apply {
            hint = "Например: мой основной проект — SWIN AI"
            setTextColor(chatColor("ink"))
            setHintTextColor(chatColor("muted"))
            setSingleLine(false)
            minLines = 2
            maxLines = 4
            textSize = 15f
            setPadding(chatDp(14), chatDp(12), chatDp(14), chatDp(12))
            background = chatBg(chatColor("panel"), 12, chatColor("stroke"))
        }
        val wrapper = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(20), chatDp(8), chatDp(20), 0)
            addView(input, LinearLayout.LayoutParams(-1, -2))
            addView(makeText(
                "Не сохраняй пароли, ключи доступа и платёжные данные.",
                11f,
                chatColor("muted")
            ).apply { setPadding(0, chatDp(8), 0, 0) })
        }
        val dialog = dialogBuilder()
            .setTitle("Новый факт памяти")
            .setView(wrapper)
            .setPositiveButton("Сохранить", null)
            .setNegativeButton("Отмена", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setTextColor(chatColor("accent"))
            dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setTextColor(chatColor("muted"))
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                val value = input.text.toString().trim()
                if (value.isBlank()) {
                    input.error = "Введи текст для сохранения"
                    return@setOnClickListener
                }
                dialog.dismiss()
                saveMemoryFact(value)
            }
        }
        dialog.show()
    }

    private fun saveMemoryFact(value: String) {
        val token = authToken ?: return
        thread {
            try {
                requestJson("/memory/facts", JSONObject().put("value", value), token)
                runOnUiThread {
                    Toast.makeText(this, "Факт сохранён в памяти SWIN", Toast.LENGTH_SHORT).show()
                    showMemorySettings()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    dialogBuilder()
                        .setTitle("Не удалось сохранить")
                        .setMessage(error.message ?: "Проверь подключение к серверу.")
                        .setPositiveButton("Повторить") { _, _ -> saveMemoryFact(value) }
                        .setNegativeButton("Закрыть", null)
                        .show()
                }
            }
        }
    }

    private fun deleteMemoryFact(memoryKey: String) {
        val token = authToken ?: return
        if (memoryKey.isBlank()) return
        memoryDialog?.dismiss()
        memoryDialog = null
        thread {
            try {
                val encoded = URLEncoder.encode(memoryKey, "UTF-8")
                requestJson("/memory/facts?memory_key=" + encoded, JSONObject(), token, method = "DELETE")
                runOnUiThread {
                    Toast.makeText(this, "Факт удалён", Toast.LENGTH_SHORT).show()
                    showMemorySettings()
                }
            } catch (error: Exception) {
                runOnUiThread {
                    Toast.makeText(this, error.message ?: "Не удалось удалить факт", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showResponseLanguageSettings() {
        val codes = arrayOf("auto", "ru", "en", "de", "es", "fr", "zh", "uk", "pt", "it", "ja")
        val labels = arrayOf(
            "Автоопределение по сообщению",
            "Русский",
            "English",
            "Deutsch",
            "Español",
            "Français",
            "中文",
            "Українська",
            "Português",
            "Italiano",
            "日本語"
        )
        val selected = codes.indexOf(responseLanguage).coerceAtLeast(0)
        dialogBuilder()
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
        dialogBuilder()
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
        dialogBuilder()
            .setTitle("Очистить историю чата?")
            .setMessage("Будут удалены сообщения этой переписки с сервера и резервная копия на телефоне. Сохранённые факты памяти SWIN останутся.")
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
                welcomePanel = null
                showWelcomePanel()
                val message = if (serverCleared) "История очищена на сервере и телефоне."
                    else "Копия на телефоне очищена. Сервер временно недоступен."
                Toast.makeText(this, message, Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun quickActionCard(
        icon: String,
        title: String,
        description: String,
        action: () -> Unit
    ): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(14), chatDp(14), chatDp(14), chatDp(13))
            background = chatBg(chatColor("panel"), 18, chatColor("stroke"))
            isClickable = true
            isFocusable = true
            minimumHeight = chatDp(124)
            elevation = chatDp(1).toFloat()
            setOnClickListener { action() }
        }
        val iconTile = TextView(this).apply {
            text = icon
            textSize = 18f
            gravity = Gravity.CENTER
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setTextColor(chatColor("accent"))
            background = chatBg(chatColor("codePill"), 11)
        }
        card.addView(iconTile, LinearLayout.LayoutParams(chatDp(36), chatDp(36)))
        card.addView(makeText(title, 14f, chatColor("ink")).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setPadding(0, chatDp(10), 0, chatDp(3))
            maxLines = 2
        })
        card.addView(makeText(description, 11f, chatColor("muted")).apply {
            setLineSpacing(chatDp(2).toFloat(), 1f)
            maxLines = 2
        })
        return card
    }

    private fun focusComposer(prompt: String, coding: Boolean = false) {
        if (coding) {
            isCodeMode = true
            updateCodingControls()
            codeModeHeader?.text = "Код ✓"
        }
        chatInput.setText(prompt)
        chatInput.setSelection(chatInput.text.length)
        chatInput.requestFocus()
        val inputMethod = getSystemService(Context.INPUT_METHOD_SERVICE) as android.view.inputmethod.InputMethodManager
        inputMethod.showSoftInput(chatInput, android.view.inputmethod.InputMethodManager.SHOW_IMPLICIT)
    }

    private fun showWelcomePanel() {
        if (!::chatMessages.isInitialized) return
        welcomePanel?.let { (it.parent as? ViewGroup)?.removeView(it) }
        val panel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(3), chatDp(30), chatDp(3), chatDp(22))
        }
        panel.addView(makeText("SWIN AI  /  WORKSPACE", 10f, chatColor("accent")).apply {
            letterSpacing = 0.15f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        panel.addView(makeText("Думай глубже.\nСоздавай точнее.", 27f, chatColor("ink")).apply {
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            setLineSpacing(chatDp(2).toFloat(), 1f)
            setPadding(0, chatDp(9), 0, chatDp(8))
        })
        panel.addView(makeText(
            "Рабочее пространство для сложной математики, разработки и технических исследований.",
            14f,
            chatColor("muted")
        ).apply {
            setLineSpacing(chatDp(3).toFloat(), 1f)
            setPadding(0, 0, 0, chatDp(20))
        })

        val firstRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        firstRow.addView(quickActionCard(
            "∑", "Сложная математика", "Алгебра, анализ и доказательства"
        ) { focusComposer("Реши задачу подробно. Обоснуй каждый шаг, проверь вычисления и выдели итоговый ответ:\n") },
            LinearLayout.LayoutParams(0, -2, 1f).apply { marginEnd = chatDp(7) })
        firstRow.addView(quickActionCard(
            "</>", "Написать код", "Python, GDScript, Kotlin и C++"
        ) { focusComposer("Разработай решение как опытный инженер. Укажи архитектуру, полный код и тесты:\n", coding = true) },
            LinearLayout.LayoutParams(0, -2, 1f).apply { marginStart = chatDp(7) })
        panel.addView(firstRow)

        val secondRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
        }
        secondRow.addView(quickActionCard(
            "⌘", "Найти ошибку", "Разбор причины и исправление"
        ) { focusComposer("Проанализируй ошибку. Найди первопричину, предложи исправление и тест для проверки:\n", coding = true) },
            LinearLayout.LayoutParams(0, -2, 1f).apply { marginEnd = chatDp(7) })
        secondRow.addView(quickActionCard(
            "∷", "Разобрать концепцию", "Алгоритмы, формулы и системы"
        ) { focusComposer("Объясни эту концепцию технически точно, с формулами и примером:\n") },
            LinearLayout.LayoutParams(0, -2, 1f).apply { marginStart = chatDp(7) })
        panel.addView(secondRow, LinearLayout.LayoutParams(-1, -2).apply {
            topMargin = chatDp(11)
        })

        panel.addView(makeText(
            "История диалогов, память и проекты доступны в меню.",
            11f,
            chatColor("muted")
        ).apply {
            setPadding(chatDp(2), chatDp(17), 0, 0)
        })
        chatMessages.addView(panel, LinearLayout.LayoutParams(-1, -2))
        welcomePanel = panel
    }

    private fun removeWelcomePanel() {
        welcomePanel?.let { (it.parent as? ViewGroup)?.removeView(it) }
        welcomePanel = null
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
        val brand = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val brandMark = TextView(this).apply {
            text = "S"
            textSize = 19f
            gravity = Gravity.CENTER
            typeface = Typeface.create("sans-serif", Typeface.BOLD)
            setTextColor(chatColor("sendText"))
            background = chatBg(accent, 13)
        }
        brand.addView(brandMark, LinearLayout.LayoutParams(chatDp(40), chatDp(40)))
        val brandCopy = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(chatDp(10), 0, 0, 0)
        }
        brandCopy.addView(makeText("SWIN AI", 18f, ink).apply {
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
        })
        brandCopy.addView(makeText("Математика · Код · Исследования", 10f, muted))
        brand.addView(brandCopy)
        brand.isClickable = true
        brand.isFocusable = true
        brand.contentDescription = "Открыть историю чатов"
        brand.setOnClickListener { showChatHistory() }
        header.addView(brand, LinearLayout.LayoutParams(0, chatDp(52), 1f))
        val codeMode = TextView(this).apply {
            text = if (isCodeMode) "Код ✓" else "Код"
            textSize = 13f
            gravity = Gravity.CENTER
            setTextColor(accent)
            setPadding(chatDp(14), 0, chatDp(14), 0)
            background = chatBg(chatColor("codePill"), 18, chatColor("stroke"))
            setOnClickListener {
                isCodeMode = !isCodeMode
                updateCodingControls()
                text = if (isCodeMode) "Код ✓" else "Код"
            }
        }
        codeModeHeader = codeMode
        val memoryButton = TextView(this).apply {
            text = "▤"
            textSize = 18f
            gravity = Gravity.CENTER
            setTextColor(ink)
            contentDescription = "Память SWIN"
            background = chatBg(chatColor("panel"), 13, chatColor("stroke"))
            setOnClickListener { showMemorySettings() }
        }
        val settings = TextView(this).apply {
            text = "⋯"
            textSize = 23f
            gravity = Gravity.CENTER
            setTextColor(ink)
            contentDescription = "Настройки"
            background = chatBg(chatColor("panel"), 13, chatColor("stroke"))
            setOnClickListener { showChatSettings() }
        }
        val themeButton = TextView(this).apply {
            text = if (isDarkTheme()) "☀" else "☾"
            textSize = 20f
            gravity = Gravity.CENTER
            setTextColor(ink)
            contentDescription = "Сменить тему"
            background = chatBg(chatColor("panel"), 13, chatColor("stroke"))
            setOnClickListener { showThemeSettings() }
        }
        header.addView(codeMode, LinearLayout.LayoutParams(-2, chatDp(38)).apply {
            marginEnd = chatDp(5)
        })
        header.addView(memoryButton, LinearLayout.LayoutParams(chatDp(38), chatDp(40)).apply {
            marginStart = chatDp(3); marginEnd = chatDp(3)
        })
        header.addView(themeButton, LinearLayout.LayoutParams(chatDp(38), chatDp(40)).apply {
            marginStart = chatDp(3); marginEnd = chatDp(3)
        })
        header.addView(settings, LinearLayout.LayoutParams(chatDp(38), chatDp(40)).apply {
            marginStart = chatDp(3)
        })
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
            adapter = object : ArrayAdapter<String>(
                this@MainActivity,
                android.R.layout.simple_spinner_item,
                langs
            ) {
                override fun getView(position: Int, convertView: View?, parent: ViewGroup): View {
                    return (super.getView(position, convertView, parent) as TextView).apply {
                        setTextColor(chatColor("ink"))
                        textSize = 14f
                        setPadding(chatDp(12), 0, chatDp(10), 0)
                    }
                }

                override fun getDropDownView(position: Int, convertView: View?, parent: ViewGroup): View {
                    return (super.getDropDownView(position, convertView, parent) as TextView).apply {
                        setTextColor(chatColor("ink"))
                        setBackgroundColor(chatColor("panel"))
                        textSize = 14f
                        setPadding(chatDp(14), chatDp(12), chatDp(12), chatDp(12))
                    }
                }
            }.also { it.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item) }
            background = chatBg(chatColor("panel"), 10, chatColor("stroke"))
        }
        projectContextInput = makeEdit("Код или структура проекта (необязательно)",
            InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE, false).apply {
            maxLines = 3
            textSize = 13f
            setTextColor(ink)
            setHintTextColor(muted)
            background = chatBg(chatColor("panel"), 12, chatColor("stroke"))
            setPadding(chatDp(12), chatDp(9), chatDp(12), chatDp(9))
        }
        activitySwitch = Switch(this).apply {
            text = "Показывать этапы проверки"
            setTextColor(muted)
            thumbTintList = ColorStateList.valueOf(accent)
            trackTintList = ColorStateList.valueOf(chatColor("stroke"))
            setOnCheckedChangeListener { _, checked -> showCodingActivity = checked }
        }
        copyCodeButton = makeButton("Скопировать последний код") {
            val cb = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            cb.setPrimaryClip(ClipData.newPlainText("SWIN AI code", lastGeneratedCode))
            Toast.makeText(this, "Код скопирован", Toast.LENGTH_SHORT).show()
        }.apply {
            isEnabled = false
            setTextColor(accent)
            background = chatBg(chatColor("codePill"), 12, chatColor("stroke"))
            stateListAnimator = null
        }
        codingPanel.addView(makeText("РЕЖИМ РАЗРАБОТКИ", 11f, accent).apply {
            letterSpacing = 0.08f
            typeface = Typeface.create("sans-serif-medium", Typeface.NORMAL)
            setPadding(chatDp(3), 0, 0, chatDp(2))
        })
        codingPanel.addView(makeText(
            "Опиши задачу и при желании добавь контекст проекта. SWIN подготовит код и покажет доступные проверки.",
            11f,
            muted
        ).apply { setPadding(chatDp(3), 0, 0, chatDp(7)) })
        val workspaceButton = TextView(this).apply {
            text = "▣  Открыть рабочее пространство"
            textSize = 12f
            gravity = Gravity.CENTER
            setTextColor(chatColor("accent"))
            background = chatBg(chatColor("codePill"), 10, chatColor("stroke"))
            setPadding(chatDp(10), 0, chatDp(10), 0)
            setOnClickListener { showWorkspace() }
        }
        codingPanel.addView(workspaceButton, LinearLayout.LayoutParams(-1, chatDp(38)).apply {
            bottomMargin = chatDp(6)
        })
        codingPanel.addView(languageSpinner, LinearLayout.LayoutParams(-1, chatDp(44)))
        codingPanel.addView(projectContextInput, LinearLayout.LayoutParams(-1, -2).apply {
            topMargin = chatDp(5)
        })
        codingPanel.addView(activitySwitch, LinearLayout.LayoutParams(-1, chatDp(40)))
        codingPanel.addView(copyCodeButton, LinearLayout.LayoutParams(-1, chatDp(42)))
        root.addView(codingPanel, LinearLayout.LayoutParams(-1, -2).apply {
            topMargin = chatDp(5); bottomMargin = chatDp(4)
        })

        chatMessages = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, chatDp(10), 0, chatDp(16))
        }
        welcomePanel = null
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
            text = "＋"
            textSize = 23f
            gravity = Gravity.CENTER
            setTextColor(muted)
            contentDescription = "Дополнительные действия"
            setOnClickListener {
                AlertDialog.Builder(this@MainActivity)
                    .setTitle("Дополнительные действия")
                    .setItems(arrayOf("История чатов", "Новый чат", "Рабочее пространство", "Режим программирования", "Память SWIN", "Язык ответа")) { _, which ->
                        when (which) {
                            0 -> openPhotoPicker()
                            1 -> showChatHistory()
                            2 -> createNewChat()
                            3 -> showWorkspace()
                            4 -> {
                                isCodeMode = !isCodeMode
                                updateCodingControls()
                                codeMode.text = if (isCodeMode) "Код ✓" else "Код"
                            }
                            5 -> showMemorySettings()
                            6 -> showResponseLanguageSettings()
                        }
                    }
                    .show()
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
        attachmentPreview = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            visibility = View.GONE
            setPadding(chatDp(4), chatDp(5), chatDp(4), chatDp(5))
        }
        attachmentScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            visibility = View.GONE
            addView(attachmentPreview, ViewGroup.LayoutParams(-2, -2))
        }
        root.addView(attachmentScroll, LinearLayout.LayoutParams(-1, chatDp(76)).apply {
            topMargin = chatDp(3)
        })
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

    private fun openPhotoPicker() {
        val intent = Intent(Intent.ACTION_OPEN_DOCUMENT).apply {
            addCategory(Intent.CATEGORY_OPENABLE)
            type = "image/*"
            putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, REQUEST_PICK_IMAGES)
        } catch (_: Exception) {
            Toast.makeText(this, "Не удалось открыть галерею.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun handlePickedImages(data: Intent) {
        val uris = mutableListOf<Uri>()
        val clip = data.clipData
        if (clip != null) {
            for (index in 0 until minOf(clip.itemCount, 3)) {
                clip.getItemAt(index).uri?.let { uris.add(it) }
            }
        } else {
            data.data?.let { uris.add(it) }
        }
        val remaining = (3 - pendingImageAttachments.size).coerceAtLeast(0)
        if (remaining == 0) {
            Toast.makeText(this, "За одно сообщение можно прикрепить до трёх фото.", Toast.LENGTH_SHORT).show()
            return
        }
        val selected = uris.distinct().take(remaining)
        if (selected.isEmpty()) return
        Toast.makeText(this, "Подготавливаю изображения…", Toast.LENGTH_SHORT).show()
        thread {
            val loaded = mutableListOf<ChatImageAttachment>()
            val failures = mutableListOf<String>()
            for (uri in selected) {
                try {
                    loaded.add(readImageAttachment(uri))
                } catch (error: Exception) {
                    failures.add(error.message ?: "неизвестная ошибка")
                }
            }
            runOnUiThread {
                pendingImageAttachments.addAll(loaded)
                renderImageAttachments()
                if (loaded.isNotEmpty()) {
                    Toast.makeText(this, "Фото прикреплено: ${loaded.size}", Toast.LENGTH_SHORT).show()
                }
                if (failures.isNotEmpty()) {
                    Toast.makeText(this, "Некоторые фото не удалось прочитать: ${failures.first()}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun readImageAttachment(uri: Uri): ChatImageAttachment {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, bounds)
        } ?: throw IllegalArgumentException("Не удалось открыть файл.")
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
            throw IllegalArgumentException("Файл не является изображением.")
        }
        var sample = 1
        while (bounds.outWidth / sample > 2048 || bounds.outHeight / sample > 2048) {
            sample *= 2
        }
        val options = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, options)
        } ?: throw IllegalArgumentException("Не удалось декодировать изображение.")
        val scaled = if (decoded.width > 2048 || decoded.height > 2048) {
            val ratio = minOf(2048f / decoded.width, 2048f / decoded.height)
            Bitmap.createScaledBitmap(
                decoded,
                (decoded.width * ratio).toInt().coerceAtLeast(1),
                (decoded.height * ratio).toInt().coerceAtLeast(1),
                true
            ).also { if (it !== decoded) decoded.recycle() }
        } else decoded

        val output = ByteArrayOutputStream()
        var quality = 86
        do {
            output.reset()
            scaled.compress(Bitmap.CompressFormat.JPEG, quality, output)
            quality -= 8
        } while (output.size() > 3_500_000 && quality >= 54)
        if (output.size() > 4_000_000) {
            scaled.recycle()
            throw IllegalArgumentException("Изображение больше 4 МБ после сжатия.")
        }
        val bytes = output.toByteArray()
        val thumbnailScale = minOf(112f / scaled.width, 112f / scaled.height, 1f)
        val thumbnail = Bitmap.createScaledBitmap(
            scaled,
            (scaled.width * thumbnailScale).toInt().coerceAtLeast(1),
            (scaled.height * thumbnailScale).toInt().coerceAtLeast(1),
            true
        )
        if (thumbnail !== scaled) scaled.recycle()
        val name = uri.lastPathSegment?.substringAfterLast('/')?.take(48) ?: "Фото"
        return ChatImageAttachment(
            "data:image/jpeg;base64," + Base64.encodeToString(bytes, Base64.NO_WRAP),
            thumbnail,
            name
        )
    }

    private fun renderImageAttachments() {
        if (!::attachmentPreview.isInitialized) return
        attachmentPreview.removeAllViews()
        attachmentPreview.visibility = if (pendingImageAttachments.isEmpty()) View.GONE else View.VISIBLE
        attachmentScroll.visibility = attachmentPreview.visibility
        for ((index, attachment) in pendingImageAttachments.withIndex()) {
            val chip = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(chatDp(5), chatDp(4), chatDp(7), chatDp(4))
                background = chatBg(chatColor("codePill"), 10, chatColor("stroke"))
            }
            chip.addView(ImageView(this).apply {
                setImageBitmap(attachment.thumbnail)
                scaleType = ImageView.ScaleType.CENTER_CROP
                contentDescription = attachment.displayName
            }, LinearLayout.LayoutParams(chatDp(52), chatDp(52)))
            chip.addView(makeText("Фото ${index + 1}", 11f, chatColor("ink")).apply {
                setPadding(chatDp(6), 0, chatDp(4), 0)
            })
            chip.addView(TextView(this).apply {
                text = "×"
                textSize = 19f
                gravity = Gravity.CENTER
                setTextColor(chatColor("muted"))
                setOnClickListener {
                    pendingImageAttachments.removeAt(index).thumbnail.recycle()
                    renderImageAttachments()
                }
            }, LinearLayout.LayoutParams(chatDp(24), chatDp(40)))
            attachmentPreview.addView(chip, LinearLayout.LayoutParams(-2, -2).apply {
                marginEnd = chatDp(7)
            })
        }
    }

    private fun activeSessionPreferenceKey(): String =
        "active_session_id:${userEmail.trim().lowercase()}"

    private fun localHistoryKey(forSessionId: String = sessionId): String =
        "chat_backup:$userEmail:$forSessionId"

    private fun readLocalHistory(forSessionId: String = sessionId): JSONArray {
        val raw = getPreferences(MODE_PRIVATE).getString(localHistoryKey(forSessionId), "[]") ?: "[]"
        return try {
            JSONArray(raw)
        } catch (_: Exception) {
            JSONArray()
        }
    }

    private fun cacheLocalMessage(
        role: String,
        content: String,
        forSessionId: String = sessionId
    ) {
        if (userEmail.isBlank() || forSessionId.isBlank() || content.isBlank()) return
        val preferences = getPreferences(MODE_PRIVATE)
        val history = readLocalHistory(forSessionId)
        history.put(
            JSONObject()
                .put("role", role)
                .put("content", content.take(12000))
        )
        while (history.length() > 60) history.remove(0)
        preferences.edit().putString(localHistoryKey(forSessionId), history.toString()).apply()
    }

    private fun createNewChat() {
        val newSession = "chat-${java.util.UUID.randomUUID()}"
        sessionId = newSession
        getPreferences(MODE_PRIVATE).edit()
            .putString(activeSessionPreferenceKey(), newSession)
            .apply()
        if (::chatMessages.isInitialized) {
            chatMessages.removeAllViews()
            welcomePanel = null
            showWelcomePanel()
        }
        if (::chatInput.isInitialized) chatInput.text.clear()
        Toast.makeText(this, "Создан новый чат", Toast.LENGTH_SHORT).show()
    }

    private fun openChatSession(targetSessionId: String) {
        if (targetSessionId.isBlank()) return
        sessionId = targetSessionId
        getPreferences(MODE_PRIVATE).edit()
            .putString(activeSessionPreferenceKey(), targetSessionId)
            .apply()
        chatMessages.removeAllViews()
        welcomePanel = null
        chatScroll.scrollTo(0, 0)
        loadChatHistory()
    }

    private fun loadChatHistory() {
        val token = authToken
        val requestedSessionId = sessionId
        if (token.isNullOrBlank() || requestedSessionId.isBlank()) {
            showWelcomePanel()
            return
        }
        thread {
            try {
                val encodedSession = URLEncoder.encode(requestedSessionId, "UTF-8")
                val response = requestJson(
                    "/memory/history?session_id=$encodedSession",
                    JSONObject(),
                    token,
                    method = "GET"
                )
                val serverMessages = response.optJSONArray("messages") ?: JSONArray()
                val localMessages = readLocalHistory(requestedSessionId)
                val restoreLocal = serverMessages.length() == 0 && localMessages.length() > 0
                val messages = if (restoreLocal) localMessages else serverMessages
                if (restoreLocal) {
                    runCatching {
                        requestJson(
                            "/memory/sync",
                            JSONObject()
                                .put("session_id", requestedSessionId)
                                .put("messages", localMessages),
                            token
                        )
                    }
                }
                runOnUiThread {
                    if (sessionId != requestedSessionId) return@runOnUiThread
                    chatMessages.removeAllViews()
                    if (messages.length() == 0) {
                        showWelcomePanel()
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
                    if (sessionId == requestedSessionId && chatMessages.childCount == 0) {
                        showWelcomePanel()
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
            if (isCodeMode) "Опишите задачу для кода…" else "Задайте сложный вопрос или опишите задачу…"
        codeModeHeader?.text = if (isCodeMode) "Код ✓" else "Код"
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
        if (message.isEmpty() && pendingImageAttachments.isEmpty()) return
        if (pendingImageAttachments.isNotEmpty()) {
            appendChatMessage(
                "assistant",
                "Фото пока не анализируются: собственная мультимодальная модель SWIN ещё не готова. " +
                    "Изображения не отправлены ни на сервер, ни стороннему провайдеру. " +
                    "Удали вложения кнопкой ×, чтобы отправить обычное сообщение."
            )
            Toast.makeText(
                this,
                "Локальное распознавание изображений пока не включено.",
                Toast.LENGTH_LONG
            ).show()
            return
        }
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
        val requestSessionId = sessionId
        chatInput.text.clear()
        removeWelcomePanel()
        appendChatMessage("user", message)
        cacheLocalMessage("user", message, requestSessionId)
        val pendingReply = appendChatMessage(
            "assistant",
            if (requestCodeMode) "Проверяю локальные инструменты…" else "Думаю…"
        )
        sendButton.isEnabled = false

        thread {
            try {
                if (requestCodeMode) {
                    val payload = JSONObject()
                        .put("task", message)
                        .put("language", selectedCodingLanguage)
                        .put("research_web", false)
                        .put("show_activity", includeCodingActivity)
                        .put("project_context", selectedProjectContext)
                        .put("session_id", requestSessionId)
                        .put("response_language", selectedResponseLanguage)
                    val response = requestJson("/code/assist", payload, token)
                    val output = StringBuilder()
                    val actualResponseLanguage = response.optString("response_language", selectedResponseLanguage)
                    val labels = codingUiLabels(actualResponseLanguage)
                    output.append("\n")
                    output.append(labels["heading"])
                    output.append("\n")
                    output.append(response.optString("summary", ""))
                    output.append("\n")
                    output.append(labels["status"])
                    output.append(": ")
                    output.append(response.optString("status", "unknown"))
                    val validation = response.optJSONObject("validation")
                    if (validation != null) {
                        output.append("\n" + (labels["syntax"] ?: "Syntax check") + ": ")
                        output.append(validation.optString("message", "не выполнена"))
                    }
                    val verification = response.optJSONObject("verification")
                    if (verification != null) {
                        output.append("\n\n" + (labels["recheck"] ?: "Verification") + ":")
                        val checkNames = listOf("research", "syntax", "cross_check", "runtime_tests")
                        val checkLabels = mapOf(
                            "research" to (labels["research"] ?: "Research"),
                            "syntax" to (labels["static"] ?: "Static analysis"),
                            "cross_check" to (labels["cross"] ?: "Cross-check"),
                            "runtime_tests" to (labels["runtime"] ?: "Runtime tests")
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
                        output.append("\n\n" + (labels["code"] ?: "CODE") + ":\n")
                        output.append(code)
                        output.append("\n")
                    } else {
                        output.append("\n")
                        output.append(labels["noCode"])
                        output.append("\n")
                    }

                    if (includeCodingActivity) {
                        val plan = response.optJSONArray("plan")
                        if (plan != null && plan.length() > 0) {
                            output.append("\n" + (labels["plan"] ?: "Work plan") + ":\n")
                            for (i in 0 until plan.length()) {
                                output.append("• ")
                                output.append(plan.optString(i))
                                output.append("\n")
                            }
                        }
                        val activity = response.optJSONArray("activity")
                        if (activity != null && activity.length() > 0) {
                            output.append("\n" + (labels["stages"] ?: "Stages") + ":\n")
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
                        output.append("\n" + (labels["sources"] ?: "Sources") + ":\n")
                        for (i in 0 until refs.length()) {
                            val item = refs.optJSONObject(i) ?: continue
                            output.append("• ")
                            output.append(item.optString("source_type", labels["webSource"] ?: "Web source"))
                            output.append(": ")
                            output.append(item.optString("title"))
                            output.append(" — ")
                            output.append(item.optString("url"))
                            output.append("\n")
                        }
                    }
                    val elapsed = response.optDouble("elapsed_ms", -1.0)
                    if (elapsed >= 0) {
                        output.append("\n" + (labels["time"] ?: "Processing time") + ": ")
                        output.append(elapsed)
                        output.append(" " + (labels["ms"] ?: "ms"))
                    }
                    val warnings = response.optJSONArray("warnings")
                    if (warnings != null) {
                        for (i in 0 until warnings.length()) {
                            output.append("\n⚠ ")
                            output.append(warnings.optString(i))
                        }
                    }
                    runOnUiThread {
                        if (sessionId == requestSessionId) {
                            pendingReply?.let { chatMessages.removeView(it) }
                        }
                        if (code.isNotBlank()) {
                            lastGeneratedCode = code
                            copyCodeButton.isEnabled = true
                        }
                        if (sessionId == requestSessionId) {
                            if (code.isNotBlank()) {
                                val codeHeader = "\n\n" + (labels["code"] ?: "CODE") + ":\n"
                                val summary = output.toString().substringBefore(codeHeader).trim()
                                appendChatMessage("assistant", summary)
                                appendChatMessage("assistant", code, true, code, selectedCodingLanguage)
                                cacheLocalMessage("assistant", summary + "\n\n[SWIN_CODE_BLOCK]\n" + code, requestSessionId)
                            } else {
                                val outputText = output.toString()
                                appendChatMessage("assistant", outputText)
                                cacheLocalMessage("assistant", outputText, requestSessionId)
                            }
                        }
                        sendButton.isEnabled = true
                    }
                } else {
                    val response = requestJson(
                        "/chat",
                        JSONObject()
                            .put("message", message)
                            .put("session_id", requestSessionId)
                            .put("response_language", selectedResponseLanguage),
                        token
                    )
                    val answer = response.optString("answer", "Сервер вернул пустой ответ.")
                    runOnUiThread {
                        if (sessionId == requestSessionId) {
                            pendingReply?.let { chatMessages.removeView(it) }
                            appendChatMessage("assistant", answer)
                            cacheLocalMessage("assistant", answer, requestSessionId)
                        }
                        sendButton.isEnabled = true
                    }
                }
            } catch (error: Exception) {
                runOnUiThread {
                    if (sessionId == requestSessionId) {
                        pendingReply?.let { chatMessages.removeView(it) }
                    }
                    if (error is ApiException && error.statusCode == 401) {
                        clearLocalSession("Сессия истекла. Войди снова.")
                    } else {
                        if (sessionId == requestSessionId) {
                            appendChatMessage("assistant", "Ошибка: ${error.message}")
                            cacheLocalMessage("assistant", "Ошибка: ${error.message}", requestSessionId)
                        }
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
