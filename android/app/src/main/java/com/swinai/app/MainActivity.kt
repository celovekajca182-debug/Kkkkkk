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
import kotlin.concurrent.thread

class MainActivity : Activity() {
    private lateinit var log: TextView
    private lateinit var input: EditText
    private lateinit var endpoint: EditText
    private lateinit var send: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12, 14, 24)
        window.navigationBarColor = Color.rgb(12, 14, 24)

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
            hint = "HTTPS-адрес своего сервера"
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine(true)
            setTextColor(Color.WHITE)
            setHintTextColor(Color.GRAY)
            setText(getPreferences(MODE_PRIVATE).getString("api_base", "") ?: "")
        }
        log = TextView(this).apply {
            text = "SWIN AI — собственный экспериментальный проект\nЛокальная база знаний работает без внешних AI API.\n"
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
            log.append("\nSWIN AI: сначала укажи HTTPS-адрес своего сервера в поле сверху.\n")
            return
        }

        getPreferences(MODE_PRIVATE).edit()
            .putString("api_base", endpoint.text.toString().trim())
            .apply()
        input.text.clear()
        log.append("\nТы: " + message + "\nSWIN AI: думаю...\n")
        send.isEnabled = false

        thread {
            val answer = try {
                val connection = URL(url).openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 15000
                connection.readTimeout = 60000
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                connection.outputStream.use {
                    it.write(JSONObject().put("message", message).toString().toByteArray(Charsets.UTF_8))
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
                "Ошибка соединения: " + error.toString()
            }
            runOnUiThread {
                log.append("SWIN AI: " + answer + "\n")
                send.isEnabled = true
            }
        }
    }
}
