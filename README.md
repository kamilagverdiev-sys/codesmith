# Codesmith

Гибридный AI-агент, который пишет код, исполняет его в изолированном Docker-sandbox, чинит свои ошибки и помнит прошлый опыт.

> **Статус:** Phase 0 + ядро Phase 1 готовы. Web UI + CLI + SSE-стрим уже работают.
> Полный план развития см. [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Что это

- **Hybrid LLM** — Claude, GPT, Ollama через один интерфейс (LiteLLM, fallback-цепочка).
- **Docker sandbox** — безопасное исполнение Python (`network: none`, cpu/mem/pids cap).
- **Self-repair loop** — агент видит stderr, правит код, повторяет.
- **Web UI** — single-page чат с **live SSE-стримом** прогресса агента (шаги, tool calls, результаты).
- **Interactive REPL** — многошаговый чат прямо в терминале.
- **CLI** — `chat`, `solve`, `repl`, `web`, `info` через Typer.
- **Memory** (Phase 2) — долговременная семантическая память через MemPalace.
- **Multi-agent** (Phase 4) — Planner / Coder / Critic / Tester.

## Quickstart — одна кнопка

На Windows достаточно дважды кликнуть по `codesmith.bat` в корне репо. Откроется меню:

```
+======================================+
|           C O D E S M I T H          |
|     hybrid AI coder + sandbox        |
+======================================+

  [1] Web UI         - browser chat with live SSE stream
  [2] REPL           - interactive multi-turn chat
  [3] Solve task     - run a task with self-repair loop
  [4] Info           - config + live health checks
  [5] Doctor         - docker / sandbox / ollama probes
  [6] Build sandbox  - rebuild docker sandbox image
  [7] Install        - bootstrap venv + dependencies
  [Q] Quit
```

`codesmith.bat` принимает и аргументы — для прямого запуска без меню:

```bat
codesmith.bat web              :: запустить Web UI и открыть браузер
codesmith.bat repl             :: интерактивный чат в терминале
codesmith.bat info             :: config + health checks
codesmith.bat doctor           :: диагностика
codesmith.bat solve "fib(20)"  :: одна задача через self-repair
```

## Quickstart — руками

### 1. Требования

- Windows 10/11 + PowerShell 5+ (или Linux/macOS — bat-launcher только под Win, остальное кросс-платформенно).
- Python **3.11+**.
- Docker Desktop с WSL2.
- Один из LLM-источников:
  - локально: **Ollama** + `qwen2.5-coder:7b` (~4.7 GB);
  - или облако: `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` в `.env`.

### 2. Установка

```powershell
git clone <this-repo> codesmith
cd codesmith
.\scripts\bootstrap.ps1 -RunTests
.\scripts\doctor.ps1
.\scripts\build-sandbox.ps1
```

Локальный стек целиком одной командой:

```powershell
.\scripts\bootstrap.ps1 -InstallOllama -InstallDocker -PullModel -Model qwen2.5-coder:7b
```

### 3. Конфигурация

```bash
cp config.example.yaml config.yaml   # primary llm + sandbox + loops
cp .env.example .env                  # секреты, в git не уходят
```

`config.yaml` по умолчанию указывает на `ollama/qwen2.5-coder:7b`. Чтобы переключиться на Claude:

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-4-6
```

(и положи `ANTHROPIC_API_KEY` в `.env`).

### 4. Запуск

| Сценарий | Команда |
|---|---|
| Web UI с live SSE-стримом | `codesmith web` или `codesmith.bat web` |
| Интерактивный чат в терминале | `codesmith repl` |
| Один вопрос — один ответ | `codesmith chat "напиши факториал"` |
| Задача с self-repair loop | `codesmith solve "fib(20) и проверь"` |
| Конфиг + live health | `codesmith info` |

## Web UI

`codesmith web` стартует FastAPI на `http://127.0.0.1:8000` и автоматически открывает браузер. UI — single-file, тёмная тема, vanilla JS, никакого build step.

Что показывает:
- **header** — текущий LLM, sandbox-лимиты, версия, индикатор живости API;
- **chat** — пузыри user/codesmith, под каждым ответом — раскрывающиеся step-карточки с tool calls и их сырыми результатами;
- **footer** — textarea, `Ctrl+Enter` для отправки, `Esc` отменяет inflight-запрос;
- **new session** — сбросить контекст и workspace.

Под капотом:
- `POST /api/sessions` — создать сессию (изолированный workspace на диске);
- `POST /api/sessions/{sid}/chat` — шлёт пользовательское сообщение, возвращает SSE-стрим:
  - `event: start` — агент принял запрос;
  - `event: step` — каждая итерация LLM с tool_calls + результатами;
  - `event: final` — финальный ответ + статистика;
  - `event: error` — фатальная ошибка, стрим закрывается;
- `GET /api/info` — конфиг без секретов, для header'а UI;
- `GET /` — статический UI.

## Структура проекта

```
codesmith/
├── codesmith.bat               ← one-click launcher (root)
├── docs/
│   ├── ROADMAP.md
│   └── ARCHITECTURE.md
├── src/codesmith/
│   ├── config.py               # YAML + .env, pydantic-валидация
│   ├── llm.py                  # LiteLLM + fallback router
│   ├── session.py              # session state + workspace
│   ├── agent.py                # agentic loop + on_step callback
│   ├── cli.py                  # Typer: chat / solve / repl / web / info
│   ├── tools/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── sandbox.py          # Docker execution (критический файл)
│   │   └── filesystem.py
│   ├── loops/
│   │   └── self_repair.py
│   └── api/
│       ├── main.py             # FastAPI: sessions + SSE chat + static
│       └── static/
│           └── index.html      # single-file Web UI
├── docker/
│   └── sandbox.Dockerfile
├── scripts/
│   ├── codesmith.ps1           # PS-обвязка для codesmith.bat
│   ├── bootstrap.ps1
│   ├── build-sandbox.ps1
│   ├── doctor.ps1
│   └── run-api.ps1
├── tests/
│   ├── test_config.py
│   ├── test_filesystem.py
│   ├── test_registry.py
│   ├── test_sandbox_integration.py
│   └── test_api.py             # FastAPI + SSE round-trip
├── config.example.yaml
├── .env.example
└── pyproject.toml
```

## Безопасность

Sandbox построен по принципу «считаем, что LLM попытается сломать систему». См. раздел 1.2 в `docs/ROADMAP.md` — все 10 правил обязательны. Если ты меняешь `tools/sandbox.py` или `docker/sandbox.Dockerfile`, прочитай этот раздел **до** коммита.

Ключевые дефолты, которые нельзя менять без обсуждения:
- `sandbox.network: none` — sandbox без сети;
- `sandbox.pids_limit: 64` — fork-bomb защита;
- `sandbox.memory_mb: 512`, `sandbox.cpus: 1.0`, `sandbox.timeout_seconds: 30`.

## Лицензия

MIT.

## Вдохновлено

- [build-your-own-openclaw](https://github.com/czl9707/build-your-own-openclaw) — паттерны LiteLLM-слоя и tool registry
- [MemPalace](https://github.com/milla-jovovich/mempalace) — долговременная память (raw mode)
- [Anthropic tool use docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use)
