# SWIN AI

Собственная экспериментальная языковая модель и Android-клиент. SWIN не обращается к ChatGPT, DeepSeek или платным AI API. Веса базовой модели создаются собственным циклом обучения с нуля.

## Архитектура SWIN 0.7

- **Unicode CharTokenizer** — русские буквы и символы кода представлены отдельными токенами; словарь сохраняется внутри checkpoint. Для старых checkpoint сохранена совместимость с UTF-8 ByteTokenizer.
- **SWINTransformer** — небольшой decoder-only Transformer с causal self-attention, позиционными embedding, feed-forward блоками и собственной матрицей весов.
- **Двухэтапное обучение** — сначала pretraining на русскоязычных документах, затем instruction tuning на вопросах и ответах.
- **Открытые данные** — FineWeb2-HQ для русского веб-текста, FineWiki для русской Wikipedia, OASST1 для проверенных диалогов и MBPP для Python-задач.
- **Фильтры данных** — проверка языка, доли кириллицы, повторов, review-флагов и некоторых нежелательных меток; маскирование очевидных email, IPv4 и телефоноподобных строк.
- **KnowledgeBase** — локальный поиск по проверенным материалам. Для известных технических вопросов retrieval остаётся приоритетнее свободной генерации.
- **Conversation memory** — история аккаунта сохраняется в PostgreSQL на хостинге или в локальном SQLite; короткий контекст сессии также используется в режиме программирования.
- **Android** — хранит уникальный session_id на телефоне, передаёт его API и продолжает поддерживать ограниченные офлайн-ответы.

## Что изучено в открытых моделях

Подробный разбор исходников, архитектур и обучающих наборов находится в [docs/research/open-models-and-datasets.md](docs/research/open-models-and-datasets.md).

В качестве референсов рассмотрены:
- [OLMo / OLMo-core](https://github.com/allenai/OLMo-core) — открытый код, обучающие рецепты и двухэтапный curriculum.
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) и [Pythia](https://huggingface.co/EleutherAI/pythia-160m) — масштабный distributed training и The Pile.
- [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) — качественный мультиязычный веб-корпус, включая русский.
- [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) — обработанные энциклопедические статьи Wikipedia; для русской версии сохраняется CC BY-SA 4.0 и атрибуция.
- [OpenAssistant OASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) — проверенные диалоговые данные.

Мы переносим подходы и явно открытые данные, а не копируем чужие веса или выдаём чужую модель за SWIN.

## Установка зависимостей

Базовый API:
\`\`\`bash
pip install -r server/requirements.txt
\`\`\`

Для обучения или запуска Transformer добавь CPU-сборку PyTorch:
\`\`\`bash
pip install -r server/requirements-transformer.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install datasets==3.3.2
\`\`\`

## Подготовка данных и обучение

Из каталога \`server\`:
\`\`\`bash
python prepare_corpus.py \
  --web-documents 2500 --web-max-chars 1500000 \
  --wiki-documents 1500 --wiki-max-chars 900000 \
  --oasst-max-pairs 3500
python prepare_coding_data.py --max-train 700 --max-validation 160
python train_transformer.py --pretrain-epochs 2 --sft-epochs 2 --block-size 512
\`\`\`

Подготовка данных потоково читает FineWeb2-HQ — отобранный русский веб-текст, FineWiki — статьи русской Wikipedia, OASST1 — проверенные диалоги. MBPP отдельно добавляет задачи по Python. Полные архивы (FineWeb2-HQ около 1,2 TB и FineWiki около 10,7 GB для русских конфигураций) не скачиваются целиком.

Скрипт создаёт:
- \`data/fineweb2_ru_train.txt\` и \`data/fineweb2_ru_validation.txt\`;
- \`data/finewiki_ru_train.txt\` и \`data/finewiki_ru_validation.txt\`;
- \`data/openassistant_ru_train.jsonl\` и \`data/openassistant_ru_validation.jsonl\`;
- \`data/mbpp_python_train.jsonl\` и \`data/mbpp_python_validation.jsonl\`;
- \`data/dataset_manifest.json\` и \`data/mbpp_manifest.json\` с источниками, лицензиями и количеством отобранных примеров.

Затем обучение выполняется в два этапа:
1. **Pretraining** — предсказание следующего Unicode-символа на смешанном корпусе FineWeb2-HQ + FineWiki + базовый текст.
2. **Instruction tuning** — обучение на вопросах и ответах из OASST1, локальной базы SWIN и программных задач MBPP.

FineWeb2, FineWiki и OASST1 имеют отдельные validation-части, которые не используются для обновления весов. Условия лицензий и атрибуция: [server/data/DATASET_SOURCES.md](server/data/DATASET_SOURCES.md).

Артефакты:
- \`models/swin-transformer-pretrain.pt\` — checkpoint после этапа предобучения;
- \`models/swin-transformer.pt\` — финальный checkpoint после instruction tuning;
- \`data/dataset_manifest.json\` — фактический состав и лицензии выборки.

GitHub Actions запускает ту же последовательность автоматически после изменений обучающего кода или исходных данных. Готовые веса и manifest публикуются в Artifacts workflow **Train SWIN Transformer**.

## Регистрация и вход

Android-клиент 0.5 включает экран регистрации и входа по email/паролю. Пароль хешируется Argon2id на сервере, а bearer-токен шифруется на телефоне ключом Android Keystore. API предоставляет `/auth/register`, `/auth/login`, `/auth/me` и `/auth/logout`; чат `/chat` и очистка памяти требуют авторизации. Электронная почта пока служит логином, подтверждение почты и восстановление пароля через письмо ещё не реализованы. Подробности: [docs/authentication.md](docs/authentication.md).

По умолчанию база аккаунтов — `server/data/swin.sqlite3`. На хостинге укажи `SWIN_DATABASE_PATH` на постоянный диск: эфемерный файловый слой может терять аккаунты при пересоздании сервиса.

## Быстродействие генерации

Сервер ограничивает число CPU-потоков через `SWIN_NUM_THREADS` (по умолчанию 2). Transformer использует causal self-attention с KV-cache: после первого prefill следующие байты используют сохранённые key/value по слоям, а rolling window пересобирается только при заполнении контекста. Генерация выполняется в `inference_mode`, API использует быстрый детерминированный greedy decoding по умолчанию. Время генерации возвращается в `generation_ms`; модель сообщает скорость сгенерированных UTF-8 байтов в секунду. Для сравнения с полной пересборкой контекста выполни из каталога `server`: `python benchmark_inference.py --checkpoint models/swin-transformer.pt --tokens 128`. Бенчмарк проверяет совпадение текста и выводит фактическое ускорение на выбранном CPU.

## Режим разработки кода

Android-клиент 0.6 добавляет режим «Разработка кода»: пользователь выбирает язык (авто, Python, Godot 4/GDScript, Kotlin, JavaScript, TypeScript, HTML, CSS или JSON) и описывает задачу. Сервер:
1. формирует краткий план реализации;
2. ищет официальную документацию через DDGS (обычный веб-поиск, не AI API);
3. передаёт задачу и короткие выдержки в собственный SWIN Transformer;
4. проверяет Python через AST, JSON через JSON-парсер; для GDScript, Kotlin, JS, TS, HTML и CSS пока есть только ограниченная проверка скобок;
5. возвращает код, результат проверки и ссылки на источники.

Подробные этапы скрыты по умолчанию. Переключатель «Показывать этапы поиска и проверки» показывает только краткие статусы и найденные источники — приватная цепочка мыслей не раскрывается. Сгенерированный код не исполняется на сервере.

**Ограничение качества:** текущая byte-level SWIN — маленькая экспериментальная модель с коротким контекстом и ограниченным обучающим корпусом. Этот режим добавляет web-research, структурированный prompt и syntax gate, но не превращает модель автоматически в полноценный Copilot. Код требует проверки человеком; наличие корректного синтаксиса не подтверждает правильность алгоритма. Для заметного роста качества нужна отдельная code-training выборка и benchmark (например, MBPP для Python) и более крупная архитектура.

API: POST /code/assist с bearer token и JSON-полями task, language, research_web, show_activity, project_context.

## Запуск API

Из каталога \`server\`:
\`\`\`bash
uvicorn app:app --host 0.0.0.0 --port 8000
\`\`\`

Основные маршруты:
- \`GET /\` и \`GET /health\` — статус и сведения о загруженных весах.
- \`POST /chat\` — JSON с \`message\` и необязательным \`session_id\`.
- \`POST /memory/clear\` — сброс истории сессии.

В Render blueprint установлен отдельный CPU PyTorch. После деплоя финальный checkpoint всё равно надо добавить в доступный серверу файл/volume и задать \`SWIN_TRANSFORMER_PATH\`, если путь отличается от стандартного.

## Android APK

Workflow **Build SWIN AI APK** собирает debug APK. Открой GitHub → Actions → успешный запуск → Artifacts и скачай \`swin-ai-debug-apk\`. В поле сверху приложения укажи HTTPS-адрес своего API. Без сервера работает встроенный офлайн-справочник, но он отвечает только на знакомые темы.

## Текущие ограничения

Это небольшая экспериментальная модель, а не большая LLM. Текущий CharTokenizer эффективнее старого ByteTokenizer для кириллицы, но остаётся символьным, а не субсловным BPE. Выборки намеренно ограничены, поскольку обучение запускается на CPU в GitHub Actions. Regex-маскирование не гарантирует удаления абсолютно всех персональных данных из веб-корпуса. Снижение validation loss само по себе не доказывает фактическую правильность ответов — для этого нужен фиксированный benchmark и человеческая оценка.

Следующие инженерные шаги: BPE/SentencePiece, preference tuning (DPO/RLHF) на корректно лицензированных предпочтениях, многоязычный корпус, расширение тестов по математике и коду, а затем длительное обучение на GPU.

SWIN AI может помогать с программированием, отладкой, своими играми, офлайн-моддингом и разрешёнными модификациями. Для собственной игры поддерживаются debug-инструменты, тестовые режимы и офлайн-трейнеры; обход античита и вмешательство в чужие онлайн-матчи не входят в проект.
