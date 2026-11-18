package com.swinai.app

internal object OfflineKnowledge {
    private data class Entry(
        val terms: List<String>,
        val answer: String,
        val minimumMatches: Int = 2
    )

    private val entries = listOf(
        Entry(
            listOf("godot", "персонаж", "движ", "characterbody3d", "move_and_slide", "gdscript"),
            "Для 3D-персонажа в Godot 4 используй CharacterBody3D. Получай ввод через Input.get_vector(), преобразуй направление относительно камеры, задавай velocity.x и velocity.z, применяй гравитацию к velocity.y и вызывай move_and_slide() в _physics_process(delta). Сам move_and_slide() вызывается без аргумента delta."
        ),
        Entry(
            listOf("parse error", "gdscript", "синтаксис", "class_name", "native class", "ошибк"),
            "При Parse Error сначала исправляй первую ошибку в Output: следующие сообщения могут быть её следствием. Проверь отступы, двоеточия после if/func, регистр имён и пути к узлам. Если class_name конфликтует с native class, дай своему классу уникальное имя."
        ),
        Entry(
            listOf("godot", "кнопк", "signal", "сигнал", "pressed"),
            "В Godot 4 сигнал кнопки можно подключить так: $Button.pressed.connect(_on_button_pressed). Обработчик объяви как func _on_button_pressed() -> void:. Проверь путь к узлу и не подключай один и тот же сигнал каждый кадр."
        ),
        Entry(
            listOf("gradle", "androidx", "gradle.properties", "daemon", "jvm"),
            "Проверь android/gradle.properties: каждое свойство должно быть на отдельной реальной строке. Например, android.useAndroidX=true — отдельная строка. Буквальные символы \\n вместо переноса могут сломать запуск Gradle. Также версия Gradle должна соответствовать Android Gradle Plugin."
        ),
        Entry(
            listOf("python", "traceback", "отлад", "ошибк", "exception"),
            "При ошибке Python изучи последнюю строку Traceback: она показывает тип ошибки. Затем найди первый кадр стека в своём файле и проверь строку, переменные и типы данных. Для диагностики используй print(type(value), repr(value)) или logging."
        ),
        Entry(
            listOf("нейросет", "обучен", "градиент", "loss", "вес"),
            "Обучение нейросети: модель делает предсказание, функция потерь вычисляет ошибку, backpropagation считает градиенты, а оптимизатор обновляет веса. Снижение loss на обучающем наборе не доказывает качество — нужна отдельная validation-выборка."
        ),
        Entry(
            listOf("transformer", "attention", "трансформер", "токенизатор"),
            "Transformer использует self-attention для моделирования связей между токенами. Для своей модели нужны токенизатор, позиционные представления, блоки attention и MLP, причинная маска, функция потерь и большой качественный корпус. Текущая SWIN RNN — только учебный первый этап."
        , 1),
        Entry(
            listOf("apk", "github actions", "артефакт", "сборк"),
            "APK собирается workflow Build SWIN AI APK. Открой GitHub → Actions → успешный запуск сборки → Artifacts и скачай swin-ai-debug-apk. Для установки на Android распакуй архив и открой APK; при необходимости разреши установку из этого источника."
        ),
        Entry(
            listOf("чит", "читы", "моддинг", "трейнер", "оффлайн", "собственн", "godot"),
            "Для собственной игры или разрешённого офлайн-моддинга могу помочь сделать debug-меню, консоль, тренировочный режим, редактор сохранений или трейнер. Напиши движок и что именно надо менять — подготовлю код и способ проверки. Не затрагиваем чужие онлайн-матчи и античит."
        )
    )

    fun answer(query: String): String? {
        val normalized = query.lowercase()
        val best = entries
            .map { entry ->
                val matches = entry.terms.count { term -> normalized.contains(term) }
                entry to matches
            }
            .maxByOrNull { it.second }
            ?: return null

        return if (best.second >= best.first.minimumMatches) best.first.answer else null
    }
}
