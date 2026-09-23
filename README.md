# Agentic RAG

Агентский RAG-сервис поверх пользовательских наборов документов. Работает
как компонент платформы: принимает вопросы через OpenAI-совместимый
Responses API, ходит в `rag_ingestion_service` за конфигурацией набора и
именами документов, ищет по OpenSearch (`kb-v2`) через гибридный BM25 +
kNN + sparse ретривал, использует Ollama (`qwen3:8b`) для переформулировки
запроса, генерации вариантов, оценки достаточности контекста и финального
ответа. Ответы стримятся клиенту через SSE, сохраняются в Postgres.

## Архитектура

```
Запрос пользователя (+ история диалога)
        │
        ▼
  ┌───────────┐
  │  Rewriter │  разворачивает местоимения/ссылки на контекст в самодостаточный запрос
  └─────┬─────┘  (вызывается один раз за весь цикл, не на каждой итерации)
        ▼
  ┌──────────────┐
  │ Multi-query  │  генерирует N альтернативных формулировок (только на 1-й итерации)
  └─────┬────────┘
        ▼
 ┌─────────────────────────────────────────┐
 │  Итерация (до max_iterations):           │
 │  ┌──────────────┐                        │
 │  │ Hybrid search│  BM25 + kNN + RRF       │
 │  └──────┬───────┘  (плоский фьюжн по     │
 │         │           всем вариантам       │
 │         ▼           запроса)             │
 │  накопление в общий пул чанков           │
 │         │                                │
 │         ▼                                │
 │  ┌──────────────────┐                    │
 │  │ Eval + Reflection │ sufficient? если  │
 │  └──────┬────────────┘ нет — next_queries│
 │         │              для след. итерации│
 │    sufficient=True ──────────────┐       │
 │    diminishing_returns ──────────┤       │
 │    max_iterations ────────────────┤       │
 └─────────────────────────────────┼───────┘
                                    ▼
                          ┌──────────────────┐
                          │  Generate answer  │  один раз, на финальном пуле чанков
                          └──────────────────┘
                                    │
                                    ▼
                          Ответ + grounded flag
```

Ключевые архитектурные решения:

- **Планировщик детерминированный**, не LLM-driven — фиксированная последовательность
  шагов вместо агента, который сам решает, какие инструменты вызывать. Осознанное
  упрощение под текущий стек моделей (gemma2:2b локально, qwen3.6:35b в проде) —
  LLM-планирование ненадёжно на некрупных моделях.
- **RRF плоский, не иерархический**: для N вариантов запроса × 2 типа поиска (BM25+kNN)
  фьюжн идёт одним проходом по всем 2N спискам сразу, а не двумя последовательными
  RRF-проходами. Первый (переписанный) запрос получает повышенный вес.
- **Eval и Reflection объединены в один LLM-вызов** (`sufficient` + `missing_aspects` +
  `next_queries` в одной структурированной JSON-схеме) — чтобы решение "хватает ли
  данных" и "что искать дальше" не расходилось между двумя независимыми вызовами.
- **Генерация ответа — один раз, в конце цикла**, не на каждой итерации — не тратим
  токены на черновики ответа, которые всё равно будут выброшены при повторном поиске.
- **Остановка цикла по трём причинам**: `sufficient` (eval решил, что данных хватает),
  `diminishing_returns` (новая итерация пересекается с уже найденным пулом на ≥80% —
  защита от бесполезного дожигания бюджета итераций), `max_iterations` (лимит исчерпан).

Сервис на FastAPI, порт `8020`. Зависимости:

- **Postgres 16** (порт `5437`) — persistence слой: диалоги, сообщения,
  источники ответов, usage, feedback.
- **OpenSearch** (порт `9200`) — гибридный поиск по индексу `kb-v2`.
- **Ollama** (порт `11434`) — LLM `qwen3:8b` для rewriter/eval/answer.
- **rag_ingestion_service** (порт `8012` внутренний, `8011` публичный) —
  резолв конфига набора и filename документов.

Схема одного хода:

```
клиент → POST /v1/responses (SSE)
  → validate model + get_conversation
  → resolve rag_config через ingestion
  → создать pending assistant message
  → SSE: response.created + response.in_progress
  → rewriter → multi_query → hybrid_search → evaluate → (?повтор)
  → answer generation
  → lookup filenames через ingestion
  → mark_ok + sources + usage (одна транзакция)
  → SSE: output_text.delta + response.completed
```

Обрыв клиента ловится через `is_disconnected()`: ход прерывается,
сообщение помечается `failed` с ошибкой `client_disconnected`.

## Быстрый старт

### 1. Поднять OpenSearch

```bash
docker compose up -d
curl http://localhost:9200/_cluster/health?pretty
```

Если контейнер падает при старте — проверить `vm.max_map_count` на хосте:
```bash
sudo sysctl -w vm.max_map_count=262144
```

Dashboards (аналог Kibana) — `http://<host>:5601`.

### 2. Установить зависимости

```bash
pip install -r requirements.txt --break-system-packages
```

Пример заполнения `.env`
```
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_USE_SSL=false
OPENSEARCH_INDEX=knowledge_base

EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_DIM=384

PASSAGE_PREFIX=""

QUERY_PREFIX="query: "
PASSAGE_PREFIX="passage: "

LLM_MODEL=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434

LLM_REQUEST_TIMEOUT=360.0
LLM_MAX_RETRIES=2
REWRITER_TEMPERATURE=0.0
MULTI_QUERY_TEMPERATURE=0.7
MULTI_QUERY_VARIANTS_COUNT=3

EVAL_TEMPERATURE=0.0
MAX_ITERATIONS=3
EARLY_STOP_OVERLAP_RATIO=0.8
```
или
```
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=qwen3.6:35b
EMBEDDING_DIM=1024
PASSAGE_PREFIX=""
QUERY_PREFIX=""
PASSAGE_PREFIX=""
```

### 3. Проверить инфраструктуру (без реальных данных)

```bash
python test_search.py --setup                 # создаёт индекс + 4 синтетических чанка
python test_search.py "как обслуживать щит"
```
```bash
curl -s http://127.0.0.1:8020/health | jq
```

### 4. Проиндексировать реальные документы

Ожидается результат парсинга MinerU в структуре:
```
<input_dir>/
    <doc_name>/
        office/
            *model.json | *content_list_v2.json | *content_list.json
            *.md
```

```bash
python create_collection_opensearch.py --input_dir ./data --reset
```

## Конфигурация (`.env`)

Всё через переменные окружения (префикс `AGENTIC_RAG_`). Полный список
с дефолтами — в `.env.example`. Ключевые:

| Переменная | Назначение | Пример |
|---|---|---|
| `AGENTIC_RAG_DATABASE_URL` | Postgres | `postgresql+asyncpg://rag:rag@localhost:5432/agentic_rag?ssl=disable` |
| `AGENTIC_RAG_OPENSEARCH_URL` | OpenSearch | `http://localhost:9200` |
| `AGENTIC_RAG_LLM_BASE_URL` | Ollama | `http://localhost:11434` |
| `AGENTIC_RAG_INGEST_INTERNAL_URL` | Internal API ingestion | `http://localhost:8012` |
| `AGENTIC_RAG_TOKENIZER_REPO` | HF-токенайзер | `Qwen/Qwen3-8B` |
| `AGENTIC_RAG_HISTORY_TOKEN_LIMIT` | Sliding window | `3000` |
| `AGENTIC_RAG_MAX_ITERATIONS` | Макс. итераций агентского цикла | `3` |
| `AGENTIC_RAG_SSE_HEARTBEAT_INTERVAL` | Секунды между heartbeat SSE | `15` |

**Про `?ssl=disable` в DATABASE_URL:** asyncpg 0.30+ по умолчанию пробует
TLS handshake, а Postgres в контейнере на него не настроен и обрывает.
Для локального stack всегда добавляй `?ssl=disable`. Для внешнего
Postgres с сертификатом — `?ssl=require`.

## API

### Аутентификация

Все ручки, кроме `/health`, требуют заголовок `X-User-Id`. Внутренний
контракт платформы: master ставит его из своей аутентификации, при
прямом обращении к сервису (dev, CLI) передавать вручную.

```
X-User-Id: 11111111-1111-1111-1111-111111111111
```

Отсутствие → **401**. Невалидный UUID → **401**.

### Формат id сообщений

Сервис принимает id сообщений в четырёх формах — они все распарсиваются
в один и тот же UUID:

- `resp_<uuid>` — от Responses API (`POST /v1/responses`, `GET /v1/responses/{id}`);
- `msg_<uuid>` — от platform listings (`GET /v1/platform/conversations/{id}/messages`);
- `chatcmpl-<uuid>` — OpenAI Chat Completions формат (для совместимости с OpenAI SDK);
- голый `<uuid>` — если клиент сохранил id без префикса.

На выход префикс зависит от endpoint'а:

- Responses API отдаёт `resp_<uuid>`;
- Platform listings — `msg_<uuid>`.

Feedback и sources ручки принимают любую форму — можно смело пересылать
id, каким его сохранил клиент.

### Health

**`GET /health`** — 200 всегда (даже при `degraded`), чтобы Kubernetes не
рестартил инстанс из-за временной недоступности зависимостей.

```bash
curl -s http://127.0.0.1:8020/health
```

```json
{
  "status": "ok",
  "ready": true,
  "opensearch": true,
  "ollama": true,
  "ingestion": true,
  "database": true
}
```

`status` = `starting` (lifespan ещё не завершил), `ok` (все up), либо
`degraded` (хоть одна зависимость down).

**`GET /health/ready`** — 200 если `state.ready` установлен, иначе 503.

---

### Responses API

Основной endpoint для генерации ответа.

#### `POST /v1/responses`

Тело:

```json
{
  "model": "rag/22222222-2222-2222-2222-222222222222",
  "input": "Как настроить SSL для внутреннего сервиса?",
  "conversation_id": "33333333-3333-3333-3333-333333333333",
  "stream": true
}
```

| Поле | Тип | Обязательно | Описание |
|---|---|:-:|---|
| `model` | `str` | ✓ | `rag/<uuid набора>` — маршрутизация в master, `<uuid>` идентифицирует конкретный набор |
| `input` | `str` | ✓ | Текст вопроса |
| `conversation_id` | UUID | ✓ | Диалог, к которому относится вопрос |
| `stream` | `bool` | — | Всегда `true` в MVP; поле оставлено для совместимости с OpenAI SDK |

Возвращает `text/event-stream` со следующей последовательностью:

```
event: response.created
data: {"id":"resp_<uuid>","object":"response","status":"in_progress",...}

event: response.in_progress
data: {"id":"resp_<uuid>","status":"in_progress"}

: ping                    ← keepalive каждые ~15 сек при долгом ходе

event: response.output_text.delta
data: {"id":"resp_<uuid>","delta":"Полный текст ответа..."}

event: response.completed
data: {"id":"resp_<uuid>","status":"completed","usage":{"prompt_tokens":...}}
```

**Пример curl:**

```bash
curl -N -X POST http://127.0.0.1:8020/v1/responses \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag/22222222-2222-2222-2222-222222222222",
    "input": "Как настроить SSL для внутреннего сервиса?",
    "conversation_id": "33333333-3333-3333-3333-333333333333"
  }'
```

Флаг `-N` в curl отключает буферизацию — иначе SSE-фреймы приходят
пачками.

**Ошибки** приходят как SSE-фрейм `response.error`, не HTTP-код (клиент
уже открыл поток и ждёт SSE):

```
event: response.error
data: {"error":{"message":"Диалог не найден","type":"not_found_error"}}
```

Возможные ошибки: конверсация не найдена, набор не найден, набор не
`ready`, `model` не совпадает с `conversation.rag_id`.

**HTTP 400** возвращается только при валидации pydantic (пустой input,
битый UUID conversation_id, невалидное тело JSON).

#### `GET /v1/responses/{response_id}`

Снапшот сохранённого ответа. Использовать когда клиент потерял
SSE-соединение или хочет проверить статус.

```bash
curl -s -X GET \
  http://127.0.0.1:8020/v1/responses/resp_44444444-4444-4444-4444-444444444444 \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

Ответ:

```json
{
  "id": "resp_44444444-4444-4444-4444-444444444444",
  "object": "response",
  "created_at": 1737123456,
  "status": "completed",
  "model": "rag/22222222-2222-2222-2222-222222222222",
  "conversation_id": "33333333-3333-3333-3333-333333333333",
  "output": [{
    "type": "message",
    "id": "msg_44444444-4444-4444-4444-444444444444",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "SSL настраивается..."}]
  }],
  "usage": {"prompt_tokens": 850, "completion_tokens": 120, "total_tokens": 970},
  "error": null
}
```

`status`: `completed` (готов), `in_progress` (внутренний `pending`),
`failed`. `output` пустой при `in_progress` или `failed` без частичного
ответа; `error` заполнен при `failed`.

---

### Platform: Conversations

CRUD над чатами.

#### `POST /v1/platform/conversations` — создать

Тело:

```json
{
  "rag_id": "22222222-2222-2222-2222-222222222222",
  "title": "Настройка SSL"
}
```

| Поле | Тип | Обязательно |
|---|---|:-:|
| `rag_id` | UUID | ✓ |
| `title` | `str` (≤500 символов) | — |

**Валидация `rag_id`:** через `rag_ingestion_service`. Набор должен
существовать и принадлежать пользователю; статус набора при этом **не**
проверяется — можно создать чат на набор в статусе `empty`/`ingesting`
(«зарезервировать» диалог до готовности документов).

```bash
curl -s -X POST http://127.0.0.1:8020/v1/platform/conversations \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{
    "rag_id": "22222222-2222-2222-2222-222222222222",
    "title": "Настройка SSL"
  }'
```

**201 Created:**

```json
{
  "id": "33333333-3333-3333-3333-333333333333",
  "rag_id": "22222222-2222-2222-2222-222222222222",
  "title": "Настройка SSL",
  "created_at": "2026-01-15T10:23:45.123456+00:00",
  "updated_at": "2026-01-15T10:23:45.123456+00:00"
}
```

Ошибки:
- **404** — набор не найден.
- **502** — `rag_ingestion_service` недоступен.
- **400** — пустое/невалидное тело, попытка передать поле не из схемы.

#### `GET /v1/platform/conversations` — список

Все чаты текущего пользователя. Порядок — `updated_at DESC` (недавно
активные наверху). Лимит фиксирован в репозитории (100).

```bash
curl -s -X GET http://127.0.0.1:8020/v1/platform/conversations \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

```json
{
  "data": [
    {
      "id": "33333333-3333-3333-3333-333333333333",
      "rag_id": "22222222-2222-2222-2222-222222222222",
      "title": "Настройка SSL",
      "created_at": "2026-01-15T10:23:45+00:00",
      "updated_at": "2026-01-15T10:25:12+00:00"
    }
  ]
}
```

#### `GET /v1/platform/conversations/{id}` — один

```bash
curl -s -X GET \
  http://127.0.0.1:8020/v1/platform/conversations/33333333-3333-3333-3333-333333333333 \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

Возвращает тот же формат, что и элемент из `list`.

**404** — не существует или принадлежит другому пользователю (не
различаем — иначе через 404-vs-403 утекает информация о чужих id).

#### `PATCH /v1/platform/conversations/{id}` — переименовать

Меняем **только** `title`. `rag_id` фиксирован при создании; попытка
передать `rag_id` → **400**.

```bash
curl -s -X PATCH \
  http://127.0.0.1:8020/v1/platform/conversations/33333333-3333-3333-3333-333333333333 \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{"title": "SSL — production"}'
```

Тело:

```json
{
  "title": "Новое название"
}
```

Пустой `title` → **400** (для очистки заголовка используй `null` при
создании; PATCH обязывает непустое значение).

#### `DELETE /v1/platform/conversations/{id}` — удалить

CASCADE удаляет messages, sources, usage, feedback.

```bash
curl -s -X DELETE \
  http://127.0.0.1:8020/v1/platform/conversations/33333333-3333-3333-3333-333333333333 \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

**204 No Content** (без тела).

#### `GET /v1/platform/conversations/{id}/messages` — история

История чата, включая failed сообщения (для UI). Cursor-based
пагинация «в сторону older»: первый вызов даёт последние N сообщений,
`before=<created_at>` подгружает более старые.

Query-параметры:

| Параметр | Тип | Дефолт | Описание |
|---|---|---|---|
| `limit` | `int` (1-200) | 50 | Максимум записей в ответе |
| `before` | ISO datetime | — | `created_at < before`; для подгрузки старее |

```bash
# Первая страница (последние 50)
curl -s -X GET \
  "http://127.0.0.1:8020/v1/platform/conversations/33333333-3333-3333-3333-333333333333/messages" \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"

# Load older: передать created_at первой видимой записи
curl -s -X GET \
  "http://127.0.0.1:8020/v1/platform/conversations/33333333-3333-3333-3333-333333333333/messages?limit=50&before=2026-01-15T10:23:45%2B00:00" \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

Ответ:

```json
{
  "data": [
    {
      "id": "msg_55555555-5555-5555-5555-555555555555",
      "object": "message",
      "role": "user",
      "content": "Как настроить SSL?",
      "status": "ok",
      "error": null,
      "created_at": "2026-01-15T10:24:00+00:00"
    },
    {
      "id": "msg_44444444-4444-4444-4444-444444444444",
      "object": "message",
      "role": "assistant",
      "content": "SSL настраивается...",
      "status": "ok",
      "error": null,
      "created_at": "2026-01-15T10:24:35+00:00"
    }
  ],
  "has_more": false
}
```

`has_more`: `true` если пришло ровно `limit` записей (возможно есть
ещё). Порядок в `data` — по `created_at ASC` (старые сверху, как в
UI-чатах).

**`status`** в MessageOut — внутренняя терминология БД (`ok`, `pending`,
`failed`). Отличается от Responses API-снапшота, где `ok → completed`.

---

### Feedback

Оценка ответа пользователем. JSONB partial update: ключи мержатся, не
перезаписывают целиком.

#### `POST /v1/chat/completions/{message_id}/feedback` — upsert

Тело — произвольный JSON-объект (min 1 ключ):

```bash
curl -s -X POST \
  http://127.0.0.1:8020/v1/chat/completions/resp_44444444-4444-4444-4444-444444444444/feedback \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "helpful": true, "tags": ["точно", "полно"]}'
```

**200 OK:**

```json
{
  "message_id": "44444444-4444-4444-4444-444444444444",
  "data": {"rating": 5, "helpful": true, "tags": ["точно", "полно"]},
  "created_at": "2026-01-15T10:25:00+00:00",
  "updated_at": "2026-01-15T10:25:00+00:00"
}
```

**Повторный POST** — merge (JSONB `||`): существующие ключи, не
упомянутые в новом теле, сохраняются:

```bash
curl -s -X POST \
  http://127.0.0.1:8020/v1/chat/completions/resp_44444444-4444-4444-4444-444444444444/feedback \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111" \
  -H "Content-Type: application/json" \
  -d '{"comment": "Отличный ответ!"}'
```

Итог: `{"rating": 5, "helpful": true, "tags": [...], "comment": "..."}`.

Ошибки:
- **400** — пустое тело `{}` (нечего мержить).
- **404** — сообщение не найдено / принадлежит другому пользователю.

#### `GET /v1/chat/completions/{message_id}/feedback`

```bash
curl -s -X GET \
  http://127.0.0.1:8020/v1/chat/completions/resp_44444444-4444-4444-4444-444444444444/feedback \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

Три возможных ответа:

- **200 + `FeedbackOut`** — feedback есть.
- **200 + `null`** — сообщение есть, feedback ещё не оставляли (важно для
  UI, чтобы показать кнопку «оценить»).
- **404** — сообщения нет / чужое.

#### `DELETE /v1/chat/completions/{message_id}/feedback`

```bash
curl -s -X DELETE \
  http://127.0.0.1:8020/v1/chat/completions/resp_44444444-4444-4444-4444-444444444444/feedback \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

**204 No Content**. Идемпотентно — второй вызов тоже 204, даже если
feedback уже удалён (а сообщение всё ещё существует).

---

### Sources

Источники ответа — какие чанки и из каких документов использовались.

#### `GET /v1/chat/completions/{message_id}/sources`

```bash
curl -s -X GET \
  http://127.0.0.1:8020/v1/chat/completions/resp_44444444-4444-4444-4444-444444444444/sources \
  -H "X-User-Id: 11111111-1111-1111-1111-111111111111"
```

**200 OK:**

```json
{
  "data": [
    {
      "order": 1,
      "chunk_id": "a1b2c3d4e5f6...",
      "document_id": "66666666-6666-6666-6666-666666666666",
      "chunk_index": 3,
      "filename": "ssl-guide.pdf"
    },
    {
      "order": 2,
      "chunk_id": "f6e5d4c3b2a1...",
      "document_id": "77777777-7777-7777-7777-777777777777",
      "chunk_index": 0,
      "filename": "(удалён)"
    }
  ]
}
```

`order` — нумерация `[N]` в тексте ответа. Клиент подсвечивает `[1]`,
`[2]`... и подтягивает соответствующий элемент.

`filename` — снимок на момент ответа: имя, которое было у документа в
`rag_ingestion_service` при завершении хода. Если документ переименуют
или удалят потом — snapshot остаётся. Плейсхолдер `(удалён)` появляется
в двух случаях:

- документ уже был удалён к моменту финализации хода;
- `rag_ingestion_service` был недоступен при попытке batch-lookup
  filename (деградация — ход всё равно завершается `ok`).

Пустой `data: []` — валидный ответ: сообщение без источников (например,
`failed` до этапа search).

**404** — сообщение не найдено / чужое.

## Типичный сценарий (end-to-end)

Полный цикл: создать чат, задать вопрос, прочитать историю, оценить
ответ.

```bash
USER=11111111-1111-1111-1111-111111111111
RAG=22222222-2222-2222-2222-222222222222

# 1. Создать чат
CONV=$(curl -s -X POST http://127.0.0.1:8020/v1/platform/conversations \
  -H "X-User-Id: $USER" \
  -H "Content-Type: application/json" \
  -d "{\"rag_id\": \"$RAG\", \"title\": \"Первый чат\"}" \
  | jq -r .id)

# 2. Задать вопрос (SSE)
curl -N -X POST http://127.0.0.1:8020/v1/responses \
  -H "X-User-Id: $USER" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"rag/$RAG\", \"input\": \"Как настроить SSL?\", \"conversation_id\": \"$CONV\"}"

# 3. Достать id ответа из SSE-потока — из response.created:
# event: response.created
# data: {"id":"resp_44444444-...", ...}
RESP=resp_44444444-4444-4444-4444-444444444444

# 4. Прочитать историю
curl -s -X GET "http://127.0.0.1:8020/v1/platform/conversations/$CONV/messages" \
  -H "X-User-Id: $USER" | jq

# 5. Прочитать источники ответа
curl -s -X GET "http://127.0.0.1:8020/v1/chat/completions/$RESP/sources" \
  -H "X-User-Id: $USER" | jq

# 6. Оценить ответ
curl -s -X POST "http://127.0.0.1:8020/v1/chat/completions/$RESP/feedback" \
  -H "X-User-Id: $USER" \
  -H "Content-Type: application/json" \
  -d '{"rating": 5, "helpful": true}' | jq
```

## Разработка

### Тесты

```bash
# Все тесты (использует SQLite in-memory + FakeIngest — без реальных БД/ollama)
python -m unittest discover -s tests -t .

# Отдельный модуль
python -m unittest tests.test_api_responses -v

# Live-тесты (требуют реальный Postgres/Ollama/OpenSearch/HF-модель)
# AGENTIC_RAG_TESTS_LIVE=1 python -m unittest tests.test_live -v
```

Тесты используют fakes (`tests/support.py`, `tests/core_fakes.py`) вместо
внешних сервисов, чтобы CI не тянул ML-модели и не требовал сети. Для
проверки реальной интеграции — см. Live-режим (2.9).

### Миграции

```bash
# Новая миграция после правки моделей
alembic revision --autogenerate -m "add tags column"
# Просмотреть сгенерированный файл в alembic/versions/, поправить при необходимости
alembic upgrade head

# Откат последней
alembic downgrade -1

# Полный сброс dev-БД (данные пропадают)
docker compose down -v && docker compose up -d
alembic upgrade head
```

### Отладочный CLI

Прогнать один запрос против живого стека, без HTTP:

```bash
python -m app.debug.query \
  --rag-id 22222222-2222-2222-2222-222222222222 \
  --query "Как настроить SSL?" \
  --top-k 5 \
  --score-threshold 0.3
```

Печатает `AgentTrace` (все стадии цикла) в pretty-JSON. Требует
запущенные Ollama, OpenSearch, ingestion. Не пишет в Postgres.
