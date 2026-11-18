package com.swinai.app

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.view.Gravity
import android.view.ViewGroup
import android.widget.*
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import kotlin.concurrent.thread

class MainActivity : Activity() {
    // Укажи HTTPS URL своего Render-сервера после публикации.
    private val apiUrl = "https://YOUR-SERVICE.onrender.com/chat"
    private lateinit var log: TextView
    private lateinit var input: EditText
    private lateinit var send: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.rgb(12,14,24)
        window.navigationBarColor = Color.rgb(12,14,24)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20,24,20,16)
            setBackgroundColor(Color.rgb(12,14,24))
        }
        val title = TextView(this).apply {
            text = "SWIN AI"; textSize = 26f; gravity = Gravity.CENTER
            setTextColor(Color.rgb(170,145,255))
        }
        log = TextView(this).apply {
            text = "Собственная экспериментальная модель\\n"
            textSize = 16f; setTextColor(Color.WHITE); setPadding(8,18,8,18)
        }
        val scroll = ScrollView(this).apply { addView(log) }
        input = EditText(this).apply {
            hint = "Сообщение..."; setTextColor(Color.WHITE); setHintTextColor(Color.GRAY)
        }
        send = Button(this).apply { text = "Отправить"; setOnClickListener { sendMessage() } }
        root.addView(title,LinearLayout.LayoutParams(-1,58))
        root.addView(scroll,LinearLayout.LayoutParams(-1,0,1f))
        root.addView(input,LinearLayout.LayoutParams(-1,ViewGroup.LayoutParams.WRAP_CONTENT))
        root.addView(send,LinearLayout.LayoutParams(-1,52))
        setContentView(root)
    }

    private fun sendMessage() {
        val message=input.text.toString().trim()
        if(message.isEmpty()) return
        input.text.clear()
        if(apiUrl.contains("YOUR-SERVICE")) {
            log.append("\\nТы: $message\\nНастрой apiUrl в MainActivity.kt.\\n"); return
        }
        log.append("\\nТы: $message\\nSWIN AI: запрос отправлен...\\n")
        send.isEnabled=false
        thread {
            val answer=try {
                val c=URL(apiUrl).openConnection() as HttpURLConnection
                c.requestMethod="POST"; c.connectTimeout=15000; c.readTimeout=60000
                c.doOutput=true; c.setRequestProperty("Content-Type","application/json")
                c.outputStream.use { it.write(JSONObject().put("message",message).toString().toByteArray()) }
                val stream=if(c.responseCode in 200..299) c.inputStream else c.errorStream
                val body=stream.bufferedReader().use { it.readText() }; c.disconnect()
                val j=JSONObject(body); j.optString("answer",j.optString("detail","Нет ответа"))
            } catch(e:Exception) { "Ошибка соединения: ${e.message}" }
            runOnUiThread { log.append("SWIN AI: $answer\\n"); send.isEnabled=true }
        }
    }
}
