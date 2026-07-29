# Agentic RAG для технической документации предприятия

Retrieval-augmented генерация с агентным циклом уточнения над корпусом технической
документации (регламенты, инструкции, положения). Вместо одного прохода "нашли —
ответили" система сама оценивает, хватает ли найденного контекста, и при необходимости
переформулирует запрос и ищет ещё — до нескольких итераций, прежде чем сгенерировать
финальный ответ.

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

## Стек

- **Backend**: Python + FastAPI *(FastAPI-обвязка ещё не реализована — см. "Что дальше")*
- **Хранилище**: OpenSearch (BM25 + kNN в одном движке, Apache 2.0 — выбран вместо
  Elasticsearch из-за требования к юридической чистоте лицензии для встраивания в
  коммерческий продукт)
- **Эмбеддинги**: `intfloat/multilingual-e5-small` по умолчанию (384 dim, через
  `sentence-transformers`), но модель и префиксы настраиваются в `.env` — так же
  использовались варианты BAAI/bge (m3 без префиксов, старые bge-v1.5 — с инструкцией
  только на стороне запроса)
- **LLM**: Ollama (`gemma2:2b` для локальной разработки, `qwen3.6:35b` в проде через
  nginx-прокси) — структурированный вывод по JSON Schema через официальный `ollama` client
- **Конфигурация**: `pydantic-settings`, все параметры и системные промпты — в `.env`

## Структура проекта

```
config.py                      # Settings (pydantic-settings), включая все system-промпты
docker-compose.yml              # OpenSearch + OpenSearch Dashboards, single-node, для теста

opensearch_client.py            # фабрика клиента OpenSearch
schema.py                       # маппинг индекса (text+russian analyzer / knn_vector)
embeddings.py                   # обёртка SentenceTransformer с настраиваемыми префиксами query/passage
hybrid_search.py                # BM25 + kNN + RRF (одиночный и multi-query варианты)
create_collection_opensearch.py # индексация MinerU JSON+MD документов в OpenSearch
context_format.py               # общее форматирование чанков в пронумерованный контекст для LLM

llm_client.py                   # клиент Ollama со structured output по pydantic-схеме
rewriter.py                     # переформулировка запроса с учётом истории диалога
multi_query.py                  # генерация альтернативных формулировок запроса
evaluation.py                   # eval + reflection одним вызовом (sufficient/next_queries)
answer.py                       # генерация финального ответа (+ флаг grounded)
agent.py                        # оркестратор цикла retrieve -> eval -> (повтор | ответ)

query_search.py                 # интерактивный CLI-поиск по существующей коллекции (без LLM-слоя)
main.py                         # интерактивный CLI для полного агентного цикла (с LLM-слоем)
test_search.py                  # smoke-тест инфраструктуры на синтетических данных

requirements.txt
.env.example
```

## Быстрый старт

### 1. Поднять OpenSearch

```bash
docker compose up -d
curl http://localhost:9200/_cluster/health?pretty
```

Если контейнер падает при старте — проверь `vm.max_map_count` на хосте:
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

### 5. Проверить retrieval отдельно (без LLM-слоя)

```bash
python query_search.py                         # интерактивный REPL
python query_search.py "как обслуживать РЩ-3"
```
В REPL: строка с `|` между вариантами запускает multi-query вместо одиночного поиска.

### 6. Прогнать полный агентный цикл

```bash
python main.py
python main.py "как обслуживать РЩ-3" --top_k 5 --max_iterations 3
```
В интерактивном режиме история диалога копится автоматически — можно проверить, как
rewriter разворачивает "а как часто это делать?" в самодостаточный запрос.

## Конфигурация (`.env`)

| Переменная | Назначение |
|---|---|
| `OPENSEARCH_HOST`, `OPENSEARCH_PORT`, `OPENSEARCH_USE_SSL` | подключение к OpenSearch |
| `OPENSEARCH_INDEX` | имя индекса (алиас поля `index_name`) |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | модель эмбеддингов и размерность вектора (менять вместе — размерность у моделей разная) |
| `QUERY_PREFIX`, `PASSAGE_PREFIX` | префиксы для эмбеддингов запроса/документа — зависят от модели (e5: `"query: "`/`"passage: "`, bge-m3: пустые, старые bge-v1.5: инструкция только у запроса) |
| `OLLAMA_BASE_URL`, `LLM_MODEL` | адрес Ollama и имя модели |
| `LLM_REQUEST_TIMEOUT`, `LLM_MAX_RETRIES` | таймаут и число ретраев при невалидном JSON от LLM |
| `REWRITER_TEMPERATURE`, `MULTI_QUERY_TEMPERATURE`, `ANSWER_TEMPERATURE`, `EVAL_TEMPERATURE` | температуры по ролям |
| `MULTI_QUERY_VARIANTS_COUNT` | сколько альтернативных формулировок генерировать |
| `MAX_ITERATIONS` | лимит итераций retrieve→eval цикла |
| `EARLY_STOP_OVERLAP_RATIO` | порог пересечения с пулом для остановки по diminishing returns |
| `REWRITER_SYSTEM_PROMPT`, `MULTI_QUERY_SYSTEM_PROMPT_TEMPLATE`, `ANSWER_SYSTEM_PROMPT`, `EVAL_SYSTEM_PROMPT` | системные промпты — правки без деплоя кода |

Многострочные промпты в `.env` — в двойных кавычках (внутри можно использовать одинарные).

## Известные ограничения и технический долг

- **RRF-score не сравним между итерациями агентного цикла.** Это ранговая метрика,
  пересчитываемая заново на каждом заходе `multi_query_hybrid_search`. Для генерации
  ответа и eval это не проблема (LLM смотрит на текст), но сортировать
  `trace.final_chunks` по `_rrf_score` как единый рейтинг — некорректно.
- **Нет дедупликации между JSON- и MD-источниками одного документа.** Один и тот же
  фрагмент нередко попадает в топ дважды — как json-чанк и как md-чанк с тем же текстом.
  В старом ChromaDB-пайплайне была семантическая дедупликация, здесь её пока нет.
- **`_demote_list_titles` — пунктуационная эвристика**, настроенная на русский
  регламентный стиль перечислений (`;` в конце пункта). MinerU иногда маркирует пункты
  списка тем же `type: "title"`, что и настоящие заголовки (отличаются только по
  `level`) — без этой эвристики каждый пункт списка становился отдельным чанком и
  портил breadcrumb. Для двойной вложенности списков (список внутри пункта списка)
  возможна потеря точности breadcrumb на внутреннем уровне (контент не теряется, теряется
  только точность навигации).
- **`eval`/`answer` иногда расходятся во мнении** на слабых моделях (наблюdалось на
  gemma2:2b: eval сказал `sufficient=True`, при этом сам заполнил `next_queries`,
  который по промпту должен быть пустым при sufficient=true; `answer` для того же
  контекста поставил `grounded=False`). Стоит перепроверить на qwen3.6:35b перед тем,
  как чинить промпты — возможно, проблема в возможностях модели, а не в формулировках.
- **FastAPI-обвязка не реализована** — сейчас весь цикл тестируется через `main.py` в
  терминале. HTTP-слой, Postgres для истории чатов и пользовательских данных — следующий
  этап.
- **Суб-агенты не реализованы** — сейчас единый агент с вызовом инструментов
  (rewriter/multi-query/eval как функции, не как отдельные агенты). Возможное развитие,
  не текущая необходимость.
