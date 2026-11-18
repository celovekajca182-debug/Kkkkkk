# SWIN AI

Собственная экспериментальная языковая модель и Android-клиент. SWIN не обращается к ChatGPT, DeepSeek или платным AI API. Веса базовой модели создаются собственным циклом обучения с нуля.

## Архитектура 0.4

- **ByteTokenizer** — переводит текст в UTF-8 байты (256 возможных токенов), поддерживает русский и Unicode без отдельного словаря.
- **SWINTransformer** — небольшой decoder-only Transformer с causal self-attention, позиционными embedding, feed-forward блоками и собственной матрицей весов.
- **Двухэтапное обучение** — сначала pretraining на русскоязычных документах, затем instruction tuning на вопросах и ответах.
- **Открытые данные** — ограниченная выборка FineWeb2-HQ \`rus_Cyrl\` и проверенные русские пары из OpenAssistant OASST1.
- **Фильтры данных** — проверка языка, доли кириллицы, повторов, review-флагов и некоторых нежелательных меток; маскирование очевидных email, IPv4 и телефоноподобных строк.
- **KnowledgeBase** — локальный поиск по проверенным материалам. Для известных технических вопросов retrieval остаётся приоритетнее свободной генерации.
- **Conversation memory** — сервер хранит до 6 пар сообщений на session_id и до 256 активных сессий в RAM; история очищается при перезапуске сервера.
- **Android** — хранит уникальный session_id на телефоне, передаёт его API и продолжает поддерживать ограниченные офлайн-ответы.

## Что изучено в открытых моделях

Подробный разбор исходников, архитектур и обучающих наборов находится в [docs/research/open-models-and-datasets.md](docs/research/open-models-and-datasets.md).

В качестве референсов рассмотрены:
- [OLMo / OLMo-core](https://github.com/allenai/OLMo-core) — открытый код, обучающие рецепты и двухэтапный curriculum.
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) и [Pythia](https://huggingface.co/EleutherAI/pythia-160m) — масштабный distributed training и The Pile.
- [FineWeb2-HQ](https://huggingface.co/datasets/epfml/FineWeb2-HQ) — качественный мультиязычный веб-корпус, включая русский.
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
python prepare_corpus.py --web-documents 1200 --web-max-chars 600000 --oasst-max-pairs 1500
python train_transformer.py --pretrain-epochs 10 --sft-epochs 4 --block-size 256
\`\`\`

Подготовка данных читает FineWeb2-HQ в streaming-режиме с ограничением объёма, затем загружает OASST1 и отбирает русские проверенные пары. Полный FineWeb2-HQ (около 1,2 TB только для русской конфигурации) не скачивается.

Скрипт создаёт:
- \`data/fineweb2_ru_train.txt\` и \`data/fineweb2_ru_validation.txt\`;
- \`data/openassistant_ru_train.jsonl\` и \`data/openassistant_ru_validation.jsonl\`;
- \`data/dataset_manifest.json\` с количеством реально отобранных документов и диалогов.

Затем обучение выполняется в два этапа:
1. **Pretraining** — обучение предсказанию следующего байта на русском веб-корпусе.
2. **Instruction tuning** — продолжение обучения на вопросах и ответах из OASST1 и нашей локальной базы.

Артефакты:
- \`models/swin-transformer-pretrain.pt\` — веса после первого этапа;
- \`models/swin-transformer.pt\` — финальные веса после instruction tuning.

GitHub Actions запускает ту же последовательность автоматически после изменений обучающего кода или исходных данных. Готовые веса и manifest доступны в Artifacts workflow **Train SWIN Transformer**.

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

Это небольшая учебная модель, а не большая LLM. ByteTokenizer неэффективен для русского текста, контекст ограничен 256 байтами, а выборка данных намеренно мала, чтобы обучение помещалось в GitHub Actions. Regex-маскирование не гарантирует удаления абсолютно всех персональных данных из веб-корпуса. Снижение validation loss само по себе не доказывает фактическую правильность ответов — для этого нужен отдельный фиксированный benchmark.

Следующие инженерные шаги: обучаемый BPE/SentencePiece токенизатор, более длинный контекст, отдельный набор задач для автоматической оценки, больше проверенных русскоязычных инструкций и перенос длительного обучения на GPU.

SWIN AI может помогать с программированием, отладкой, своими играми, офлайн-моддингом и разрешёнными модификациями. Для собственной игры поддерживаются debug-инструменты, тестовые режимы и офлайн-трейнеры; обход античита и вмешательство в чужие онлайн-матчи не входят в проект.
