# Codesmith

Гибридный AI-агент, который пишет код, исполняет его в изолированном sandbox, чинит свои ошибки и помнит прошлый опыт.

> **Статус:** ранняя разработка. Phase 0 (подготовка) и ядро Phase 1 (MVP coder agent).
> Полный план развития см. [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Что это

- **Hybrid LLM** — Claude, GPT, Ollama через один интерфейс (LiteLLM).
- **Docker sandbox** — безопасное исполнение произвольного Python-кода.
- **Self-repair loop** — агент видит ошибку, правит код, повторяет.
- **Memory** (Phase 2) — долговременная семантическая память через MemPalace.
- **HTTP API + streaming** (Phase 3) — сервис вместо CLI.
- **Multi-agent** (Phase 4) — Planner / Coder / Critic / Tester.

## Quickstart

### Fast Start On Windows

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1 -RunTests
.\scripts\doctor.ps1
.\scripts\build-sandbox.ps1
```

If you want to prepare a local-only stack as well:

```powershell
.\scripts\bootstrap.ps1 -InstallOllama -InstallDocker
```

After Ollama is installed, you can pull the default local coder model:

```powershell
.\scripts\bootstrap.ps1 -PullModel -Model qwen2.5-coder:7b
```

### 1. Требования

- Python 3.11+
- Docker Desktop + WSL2 (для sandbox на Windows)
- Один из API ключей: Anthropic / OpenAI / локальная Ollama

### 2. Установка

```bash
git clone <this-repo> codesmith
cd codesmith

# uv — быстрый менеджер пакетов (или используй pip/poetry)
pip install uv
uv venv
source .venv/bin/activate

uv pip install -e ".[dev]"
```

### 3. Конфигурация

```bash
cp config.example.yaml config.yaml
cp .env.example .env
# открой .env и вставь свой ANTHROPIC_API_KEY (или OPENAI_API_KEY)
```

### 4. Собрать sandbox image

```bash
docker build -t codesmith-sandbox:latest -f docker/sandbox.Dockerfile docker/
```

### 5. Smoke test

```bash
# Простой чат
codesmith chat "напиши функцию, которая считает факториал"

# С self-repair loop — агент сам запустит код и исправит, если что
codesmith solve "напиши fibonacci(n) и проверь на n=10"
```

## Структура проекта

```
codesmith/
├── docs/
│   ├── ROADMAP.md          ← главный документ, читать первым
│   └── ARCHITECTURE.md     ← технические детали
├── src/codesmith/
│   ├── config.py           # загрузка YAML + .env
│   ├── llm.py              # обёртка над LiteLLM
│   ├── session.py          # session state
│   ├── agent.py            # основной agentic loop
│   ├── cli.py              # Typer CLI
│   ├── tools/
│   │   ├── base.py         # интерфейс Tool
│   │   ├── registry.py     # регистр инструментов
│   │   ├── sandbox.py      # Docker execution (критический файл)
│   │   └── filesystem.py   # read/write с изоляцией workspace
│   ├── loops/
│   │   └── self_repair.py  # write → run → error → fix
│   └── api/
│       └── main.py         # FastAPI (Phase 3)
├── docker/
│   └── sandbox.Dockerfile  # минимальный образ для исполнения кода
├── scripts/
│   ├── bootstrap.ps1       # разворачивает локальное окружение
│   ├── build-sandbox.ps1   # собирает docker image для sandbox
│   ├── doctor.ps1          # проверяет готовность машины
│   └── run-api.ps1         # запускает FastAPI сервер
├── tests/
├── config.example.yaml
├── .env.example
└── pyproject.toml
```

## Безопасность

Sandbox построен по принципу «считаем, что LLM попытается сломать систему». См. раздел 1.2 в `docs/ROADMAP.md` — все 10 правил обязательны. Если ты меняешь `sandbox.py`, прочитай этот раздел **до** того, как коммитить.

## Лицензия

MIT.

## Вдохновлено

- [build-your-own-openclaw](https://github.com/czl9707/build-your-own-openclaw) — паттерны LiteLLM-слоя и tool registry
- [MemPalace](https://github.com/milla-jovovich/mempalace) — долговременная память (raw mode)
- [Anthropic tool use docs](https://docs.claude.com/en/docs/agents-and-tools/tool-use)
