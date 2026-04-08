# Architecture

Технические детали, которые не помещаются в ROADMAP. Обновляется параллельно с кодом.

## Слои

```
┌──────────────────────────────────────────────────┐
│  Entry points (CLI / HTTP API / MCP)             │  ← Phase 0, 3
├──────────────────────────────────────────────────┤
│  Agent / Session                                  │  ← Phase 1
├──────────────────────────────────────────────────┤
│  Loops (self-repair, compaction)                  │  ← Phase 1, 2
├──────────────────────────────────────────────────┤
│  Tools (sandbox, fs, memory, web)                  │  ← Phase 1, 2
├──────────────────────────────────────────────────┤
│  LLM Layer (LiteLLM wrapper)                       │  ← Phase 0
├──────────────────────────────────────────────────┤
│  Storage (SQLite sessions, ChromaDB via MemPalace) │  ← Phase 2
└──────────────────────────────────────────────────┘
```

Правило: верхние слои знают про нижние, нижние про верхние — нет.

## Ключевые абстракции

### LLMProvider

Один класс, обёртка над `litellm.acompletion`. Принимает конфиг, отдаёт унифицированный ответ. Переключение провайдера — через `config.yaml`, не через код.

### Tool

```python
class Tool(Protocol):
    name: str
    description: str
    parameters: dict  # JSON Schema

    async def execute(self, session: Session, **kwargs) -> ToolResult: ...
```

Схема инструмента отдаётся в LLM в формате OpenAI function calling. LiteLLM переводит в формат конкретного провайдера.

### ToolResult

```python
@dataclass
class ToolResult:
    ok: bool
    content: str          # что увидит LLM в tool_result
    metadata: dict        # для логов/телеметрии
    error: str | None
```

### Session

Держит:
- `session_id`
- `messages: list[Message]` — история в формате LiteLLM
- `workspace_dir: Path` — изолированная рабочая папка этой сессии
- `metadata: dict`

Один session = один workspace. Это важно: две сессии не должны драться за файлы.

### Agent

Orchestrator верхнего уровня:
- хранит tool registry
- запускает agentic loop: `llm.chat(tools=...) → parse tool_calls → execute → append → repeat`
- лимитирует итерации

### Loop (мета-логика)

Loop — это policy поверх Agent. Примеры:
- `SelfRepairLoop` — крутит агент, пока задача не решена или не истёк бюджет попыток
- `CompactionLoop` — обрабатывает длинные истории

Loop не знает о LLM напрямую, только через Agent.

## Потоки данных

### Синхронный вызов

```
CLI.chat("реши X")
  → Agent.solve(task="X")
    → SelfRepairLoop.run(task)
      loop:
        → Agent.step(messages + tools)
          → LLM.chat(messages, tools=[...])  # LiteLLM
          ← AssistantMessage(tool_calls=[...])
        → execute each tool_call
          → Sandbox.execute(code)
            → docker run ...
          ← ToolResult(ok/err, stdout, stderr)
        → append tool_result to messages
      ← решение или fail
    ← result
  ← текст для юзера
```

### Потоковый вызов (Phase 3)

То же, но Agent.step — async generator, отдаёт события:
- `TextDelta(text)` — кусок ответа
- `ToolCallStarted(name, args)`
- `ToolCallCompleted(result)`
- `Done(usage)`

API-сервер перекодирует эти события в SSE.

## Где живут данные

```
$HOME/.codesmith/
├── sessions.db           # SQLite: сессии + сообщения
├── palace/               # MemPalace ChromaDB (Phase 2)
├── workspaces/           # изолированные рабочие папки сессий
│   └── <session_id>/
├── logs/
│   └── agent.jsonl
└── config.yaml           # симлинк или копия
```

Всё изолировано под `~/.codesmith`. Это упрощает backup и удаление.

## Конфиг: одна точка правды

Всё управление поведением — через `config.yaml`. Hardcoded константы в коде — только те, что не имеют смысла в конфиге (названия таблиц и т.п.).

```yaml
llm:
  provider: anthropic
  model: claude-opus-4-6
  temperature: 0.3
  max_tokens: 8000

sandbox:
  image: codesmith-sandbox:latest
  timeout_seconds: 30
  memory_mb: 512
  cpus: 1.0
  network: none

loops:
  self_repair:
    max_attempts: 5
  compaction:
    trigger_at_messages: 50
    trigger_at_tokens: 20000

memory:
  enabled: true
  palace_path: ~/.codesmith/palace
  mode: raw  # НЕ aaak, НЕ rooms — raw даёт лучший retrieval

tools:
  filesystem:
    enabled: true
  web_search:
    enabled: false  # включить в Phase 2
    provider: tavily
    max_calls_per_session: 5
```

## Что НЕ в архитектуре (сознательно)

- **Нет LangChain abstractions** — LiteLLM + свой тонкий слой. Нам не нужны Runnables, Chains, Memory классы с магией.
- **Нет очередей сообщений (Kafka/Redis)** на Phase 1-3 — in-memory достаточно. Вернёмся, если реально упрёмся.
- **Нет микросервисов** — один процесс с async. Разделение на сервисы — антипаттерн, пока у тебя нет команды из 5+ человек.
- **Нет GraphQL** — REST хватит.
- **Нет event sourcing** — простые таблицы, обычные CRUD.

## Точки расширения (на будущее)

Где заложены hooks для новых фич:

- **LLMProvider.from_config** — можно подключить новый провайдер через LiteLLM, код трогать не нужно.
- **Tool.register()** — новый инструмент = один файл в `tools/`, никаких регистраций в оркестраторе.
- **Loop.run()** — новый цикл (plan-act, reflexion и т.п.) = один файл в `loops/`.
- **Event bus** (Phase 3) — новый канал (Telegram, Slack) = один adapter, остальное переиспользуется.

## Зависимости между файлами

```
cli.py      → agent.py → llm.py
            → session.py
            → tools/*
api/main.py → agent.py (тот же)
            → session.py
loops/*     → agent.py
tools/*     → session.py (только Read)
```

Циклических зависимостей быть не должно. `tools/` не импортирует `loops/`, `loops/` не импортирует `api/`.
