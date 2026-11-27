# Docker Compose

```sh
docker compose config --quiet
docker compose up -d --build --wait
python -m tools.docker_smoke
docker compose exec -T worker python -m tools.mcp_smoke
```

Панель открывается на `127.0.0.1:8080`. `EVO_WEB_PORT` меняет только опубликованный порт. Для UNSW положи официальные CSV в `data/unsw` до запуска worker. После замены файлов перезапусти worker и metrics, чтобы пересоздать копию train и кэши.

| Сервис | Назначение | Пределы |
|---|---|---:|
| web | Nginx на loopback хоста | 0.5 CPU, 128 MiB |
| worker | HTTP-панель, эволюция и отчёты | 2 CPU (по умолчанию), 4 GiB |
| metrics | MCP Streamable HTTP во внутренней сети | 1 CPU, 1 GiB |

Лимит worker можно изменить через `EVO_WORKER_CPUS` в окружении Compose или `.env`, например `EVO_WORKER_CPUS=4.0` при наличии не менее четырёх CPU, доступных Docker.

Сервисы работают без root. Worker читает датасеты из read-only bind mount и атомарно копирует официальный train в том `metrics-train`. Metrics читает этот том и историю в read-only режиме. Worker получает только перечисленные параметры Groq из окружения или Compose `.env`. Существующий `.env` сохраняется локально.

Результаты и база квот лежат в существующих томах `results` и `state`. Для импорта старой локальной истории используй `docker compose --profile migration run --rm migrate`. Команда копирует отсутствующие записи, оставляя существующие файлы назначения. `docker compose restart worker` возвращает доступ к архиву, отчётам и журналам. Прерванный запуск восстанавливается до последнего снимка для просмотра. Автоматическое продолжение с восстановлением RNG не реализовано. Обычная остановка панели ждёт ближайшей границы порции вычислений.

Shinka устанавливается отдельно через `requirements-shinka.txt` или при сборке worker с `INSTALL_SHINKA=1`. Запуск предполагает изолированное исследовательское окружение и явный бюджет вызовов Groq. Основные Docker-проверки используют Local и проходят без внешнего LLM.
