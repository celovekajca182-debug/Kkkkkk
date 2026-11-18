package com.swinai.app

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.text.InputType
import android.view.Gravity
import android.view.ViewGroup
import android.widget.*
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var log: TextView
    private lateinit var input: EditText
    private lateinit var endpoint: EditText
    private lateinit var send: Button
    private lateinit var sessionId: String

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12, 14, 24)
        window.navigationBarColor = Color.rgb(12, 14, 24)

        val preferences = getPreferences(MODE_PRIVATE)
        sessionId = preferences.getString("session_id", null) ?: UUID.randomUUID().toString()
        preferences.edit().putString("session_id", sessionId).apply()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 18, 20, 14)
            setBackgroundColor(Color.rgb(12, 14, 24))
        }
        val title = TextView(this).apply {
            text = "SWIN AI"
            textSize = 27f
            gravity = Gravity.CENTER
            setTextColor(Color.rgb(170, 145, 255))
        }
        endpoint = EditText(this).apply {
            hint = "HTTPS-адрес своего сервера (необязательно)"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(true)
            setTextColor(Color.WHITE)
            setHintTextColor(Color.GRAY)
            setText(preferences.getString("api_base", "") ?: "")
        }
        log = TextView(this).apply {
            text = "SWIN AI 0.3 — Transformer + память диалога\\nОфлайн-справочник включён.\\n"
            textSize = 16f
            setTextColor(Color.WHITE)
            setPadding(8, 16, 8, 16)
        }
        val scroll = ScrollView(this).apply { addView(log) }
        input = EditText(this).apply {
            hint = "Напиши сообщение..."
            setTextColor(Color.WHITE)
            setHintTextColor(Color.GRAY)
        }
        send = Button(this).apply {
            text = "Отправить"
            setOnClickListener { sendMessage() }
        }

        root.addView(title, LinearLayout.LayoutParams(-1, 54))
        root.addView(endpoint, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(scroll, LinearLayout.LayoutParams(-1, 0, 1f))
        root.addView(input, LinearLayout.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(send, LinearLayout.LayoutParams(-1, 52))
        setContentView(root)
    }

    private fun getChatUrl(): String {
        val base = endpoint.text.toString().trim().trimEnd('/')
        return if (base.endsWith("/chat")) base else base + "/chat"
    }

    private fun sendMessage() {
        val message = input.text.toString().trim()
        if (message.isEmpty()) return

        val url = getChatUrl()
        if (!url.startsWith("https://") || url.contains("....")) {
            log.append("\\nТы: " + message + "\\n")
            val offlineAnswer = OfflineKnowledge.answer(message)
            if (offlineAnswer != null) {
                log.append("SWIN AI (офлайн): " + offlineAnswer + "\\n")
            } else {
                log.append("SWIN AI: в офлайн-справочнике пока нет подходящего материала. Подключи HTTPS-сервер для Transformer и памяти диалога.\\n")
            }
            input.text.clear()
            return
        }

        getPreferences(MODE_PRIVATE).edit()
            .putString("api_base", endpoint.text.toString().trim())
            .apply()
        input.text.clear()
        log.append("\\nТы: " + message + "\\nSWIN AI: думаю...\\n")
        send.isEnabled = false

        thread {
            val answer = try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 15000
                connection.readTimeout = 60000
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                val payload = JSONObject()
                    .put("message", message)
                    .put("session_id", sessionId)
                connection.outputStream.use {
                    it.write(payload.toString().toByteArray(Charsets.UTF_8))
                }
                val code = connection.responseCode
                val stream = if (code in 200..299) connection.inputStream else connection.errorStream
                val body = stream?.bufferedReader(Charsets.UTF_8)?.use { it.readText() } ?: ""
                connection.disconnect()
                val json = JSONObject(body)
                if (code in 200..299) {
                    json.optString("answer", "Сервер вернул пустой ответ.")
                } else {
                    json.optString("detail", "Ошибка сервера HTTP " + code)
                }
            } catch (error: Exception) {
                val offlineAnswer = OfflineKnowledge.answer(message)
                if (offlineAnswer != null) {
                    "Сервер недоступен — офлайн-ответ:\\n" + offlineAnswer
                } else {
                    "Ошибка соединения: " + error.toString()
                }
            }
            runOnUiThread {
                log.append("SWIN AI: " + answer + "\\n")
                send.isEnabled = true
            }
        }
    }
}
