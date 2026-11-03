# Items Description

Небольшой FastAPI-сервис для генерации черновиков B2B-карточек товаров на основе предоставленных характеристик и LLM.

Проект реализуется поэтапно. На первом этапе доступен только `GET /healthz`; генерация карточек будет добавлена позднее.

## Требования

- Python 3.12 или новее
- pip

## Локальная установка (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Если используешь другую поддерживаемую версию Python, замени `py -3.12` соответствующей командой.

## Проверки

```powershell
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pyright
```

## Запуск

```powershell
python -m uvicorn app.main:app --reload
```

После запуска:

- [Healthcheck](http://127.0.0.1:8000/healthz) — `{"status":"ok"}`
- [Swagger UI](http://127.0.0.1:8000/docs)

## Текущий статус

Этап 1: базовая структура и проверка работоспособности API. LLM, товарные схемы и бизнес-логика пока не реализованы.
