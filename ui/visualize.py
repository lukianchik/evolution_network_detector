"""Создание автономного HTML-отчёта по эволюции без дополнительных библиотек."""

import argparse
import difflib
import html
import json
from collections import defaultdict
from pathlib import Path

from artifacts.history_store import candidate_path, has_history, read_history_payload, selected_row
from integrations.usage import read_usage


def load_history(folder: Path, payload: dict | None = None) -> tuple[str, list[dict], int]:
    """Считывает результаты автономного эксперимента или запуска ShinkaEvolve."""
    if payload is not None or has_history(folder):
        data = payload if payload is not None else read_history_payload(folder)
        return data.get("mode", "unknown"), data["evaluations"], data.get("novelty_rejected", 0)
    file = folder / "history.jsonl"
    if file.exists():
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return "shinka", rows, 0
    raise FileNotFoundError(f"Не найдена история запусков в {folder}")


def line_chart(rows: list[dict], key: str, name: str, color: str) -> str:
    """Рисует изменения метрики каждого кандидата и лучший достигнутый результат."""
    width, height = 870, 260
    left, top, right, bottom = 55, 15, 30, 44
    w, h = width-left-right, height-top-bottom
    minimum = min(-0.2, *(float(row[key]) for row in rows)) if key == "fitness" and rows else 0.0
    maximum = 1.0
    values = [float(row[key]) for row in rows]
    if not values:
        return ""
    records = []
    current = float("inf") if key == "fpr" else -float("inf")
    for value in values:
        current = min(current, value) if key == "fpr" else max(current, value)
        records.append(current)

    def point(index: int, value: float) -> tuple[float, float]:
        """Преобразует номер оценки и значение метрики в координаты SVG."""
        x = left + (index / max(1, len(rows)-1))*w
        y = top + (maximum - max(minimum, min(maximum, value))) / (maximum-minimum)*h
        return x, y

    def polyline(series: list[float]) -> str:
        """Соединяет оценки линией в порядке вычисления."""
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(i, v) for i, v in enumerate(series)))

    axes = []
    for i in range(5):
        value = minimum + (maximum-minimum)*(4-i)/4
        y = point(0, value)[1]
        axes.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+w}" y2="{y:.1f}" stroke="#344157" stroke-dasharray="4 5" />')
        axes.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" fill="#9ba9bd" font-size="12">{value:.2f}</text>')
    circles = "".join(
        f'<a href="#record-{html.escape(str(rows[i]["id"]))}"><circle cx="{point(i, value)[0]:.1f}" cy="{point(i, value)[1]:.1f}" r="4" fill="#79a9ff">'
        f'<title>{html.escape(str(rows[i]["id"]))}: {value:.3f}</title></circle></a>'
        for i, value in enumerate(values)
    )
    label = 'Чем ниже, тем лучше' if key == 'fpr' else 'Чем выше, тем лучше'
    return (f'<section class="panel"><h2>{html.escape(name)}</h2>'
            f'<p class="muted">{label}. Пунктир — рекорд по этой метрике; синяя линия — результат очередной программы. Нажми на точку для просмотра кода.</p>'
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Динамика {html.escape(name)}">'
            f'{"".join(axes)}'
            f'<polyline points="{polyline(values)}" fill="none" stroke="#79a9ff" stroke-width="2.5" />'
            f'<polyline points="{polyline(records)}" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="7 5" />'
            f'{circles}</svg><p class="muted">Оценено программ: {len(rows)}</p></section>')


def code_diff(before: str, after: str, old_name: str, new_name: str) -> str:
    """Показывает удалённые и добавленные строки с подсветкой для HTML-отчёта."""
    changes = difflib.unified_diff(before.splitlines(), after.splitlines(),
                                   fromfile=old_name, tofile=new_name, lineterm="")
    lines = []
    for line in changes:
        style = ("delta-meta" if line.startswith(("@@", "+++", "---")) else
                 "delta-plus" if line.startswith("+") else
                 "delta-minus" if line.startswith("-") else "")
        lines.append(f'<span class="delta {style}">{html.escape(line)}</span>')
    return '<pre class="code-diff">' + ("".join(lines) if lines else "Изменений нет") + '</pre>'


def best_timeline(rows: list[dict]) -> str:
    """Показывает только те программы, которые обновили рекорд по fitness."""
    best = -float("inf")
    items = []
    for row in rows:
        if float(row.get("selection_score", row["fitness"])) > best:
            best = float(row.get("selection_score", row["fitness"]))
            items.append('<a class="milestone" href="#record-'+html.escape(str(row["id"]))+'">'
                         '<strong>'+html.escape(str(row["id"]))+'</strong>'
                         '<small>Поколение '+str(row["generation"])+ '</small>'
                         '<b>F1 '+f'{row["f1"]:.3f}'+'</b>'
                         '<small>Fitness '+f'{best:.3f}'+'</small></a>')
    return ('<section class="panel"><h2>Эволюция лидера: от Seed до лучшей версии</h2>'
            '<p class="muted">Каждый блок — новый рекорд по выбранной цели, а не любая сгенерированная версия. '
            'Нажми на блок, чтобы посмотреть изменения кода.</p>'
            '<div class="milestones">' + ''.join(items) + '</div></section>')


def lineage_chart(rows: list[dict]) -> str:
    """Отображает реальные связи родитель–потомок для локального и API-режимов."""
    by_generation = defaultdict(list)
    for row in rows:
        by_generation[row["generation"]].append(row)
    generations = sorted(by_generation)
    if len(rows) > 150:
        return '<p>Для большого архива дерево скрыто; используй таблицу кандидатов ниже.</p>'
    width = 1040
    height = max(300, max(len(group) for group in by_generation.values())*90+90)
    coords = {}
    for column, generation in enumerate(generations):
        group = by_generation[generation]
        for index, row in enumerate(group):
            coords[row["id"]] = (75 + column*(width-150)/max(1,len(generations)-1),
                                 48+(index+0.5)*(height-105)/len(group))
    edges = []
    nodes = []
    for row in rows:
        x,y = coords[row["id"]]
        for parent, dashed in ((row.get("parent"), False), (row.get("second_parent"), True)):
            if parent in coords:
                px,py=coords[parent]
                edges.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#60738d" stroke-width="1.5" {"stroke-dasharray=\"4 5\"" if dashed else ""} />')
        fraction=max(0,min(1,row["fitness"]))
        # Цветовая шкала показывает fitness; значение подписано в каждой точке.
        color = "#3cb8aa" if fraction>=0.6 else "#cfa55a" if fraction>=0.3 else "#8c98ac"
        tip=html.escape(f'{row["id"]} | fitness {row["fitness"]:.3f} | F1 {row["f1"]:.3f}')
        nodes.append(f'<a href="#record-{html.escape(row["id"])}"><circle cx="{x:.1f}" cy="{y:.1f}" r="18" fill="{color}" stroke="#182337" stroke-width="2"><title>{tip}</title></circle><text x="{x:.1f}" y="{y+4:.1f}" font-size="11" fill="#111827" text-anchor="middle" pointer-events="none">{html.escape(row["id"][-3:])}</text></a>')
    labels=''.join(f'<text x="{75+i*(width-150)/max(1,len(generations)-1):.1f}" y="{height-10}" text-anchor="middle" fill="#a5b4ca" font-size="14">Пок. {g}</text>' for i,g in enumerate(generations))
    return f'<section class="panel"><h2>Дерево эволюции</h2><p class="muted">Каждая точка — оценённая версия кода. Линии показывают родителей, когда данные о них записаны (Shinka хранит свой отдельный архив). Нажми на точку для просмотра версии.</p><div class="scroll"><svg viewBox="0 0 {width} {height}" style="min-width:650px" role="img" aria-label="Граф происхождения программ">{"".join(edges)}{"".join(nodes)}{labels}</svg></div></section>'


def pareto_chart(rows, saved, stats):
    """Показывает фронт, распределение попыток и параметры выборки."""
    frontier = set(saved.get("pareto_ids", ()))
    if not frontier:
        return ""
    circles = []
    for row in rows:
        if row["id"] in frontier:
            x, y = 45 + row["fpr"] * 470, 168 - row["f1"] * 150
            color = "#4ad5c2" if row["id"] == saved.get("winner_id") else "#79a9ff"
            circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}">'
                           f'<title>{html.escape(str(row["id"]))}: F1 {row["f1"]:.3f}, FPR {row["fpr"]:.3f}</title></circle>')
    parts = stats.get("sample", {})
    strategies = " · ".join(f"{html.escape(str(name))} {count}" for name, count in
                            stats.get("strategy_attempts", {}).items())
    return ('<section class="panel"><h2>Фронт Pareto</h2>'
            '<svg viewBox="0 0 550 200" role="img" aria-label="Точки F1 и FPR фронта Pareto">'
            '<path d="M45 18 V168 H515" fill="none" stroke="#9ba9bd"/>'
            '<text x="15" y="28" fill="#9ba9bd">F1</text><text x="480" y="190" fill="#9ba9bd">FPR</text>'
            '<text x="32" y="186" fill="#9ba9bd">0</text><text x="506" y="164" fill="#9ba9bd">1</text>'
            + "".join(circles) + '</svg>'
            f'<p>Стратегии · {strategies}</p>'
            f'<p>Train {parts.get("train_size", "—")} · Validation {parts.get("validation_size", "—")} · '
            f'Проверки train {stats.get("train_evaluations", 0)} · '
            f'агрегированные проверки ошибок {stats.get("error_stat_checks", 0)} · '
            f'Validation {stats.get("validation_evaluations", 0)}</p>'
            f'<p class="muted">Выборка {html.escape(str(parts.get("id", "—")))}</p></section>')


def create_report(folder: Path) -> Path:
    """Создаёт готовый к открытию report.html с графиками, схемой и кодом."""
    folder = Path(folder).resolve()
    saved = read_history_payload(folder) if has_history(folder) else None
    mode, rows, rejected = load_history(folder, saved)
    if not rows:
        raise ValueError("Нет результатов для визуализации")
    dataset_file = folder / "dataset.txt"
    dataset = (dataset_file.read_text(encoding="utf-8").strip()
               if dataset_file.is_file() else "synthetic")
    data_label = ("UNSW-NB15 (исследовательский набор сетевых потоков)"
                  if dataset == "unsw" else "синтетических размеченных событиях")
    first = rows[0]
    saved = saved or {}
    best = selected_row(rows, saved)
    label = {"demo":"Демо: программные мутации (без LLM)","openrouter":"OpenRouter: реальные LLM-мутации",
             "shinka":"ShinkaEvolve: LLM-мутации", "groq":"Groq: LLM-мутации",
             "local":"Локальная эволюция правил без LLM",
             "hybrid":"Гибридная эволюция правил с Groq",
             "hybrid-groq":"Гибридная эволюция: локальный поиск + Groq",
             "hybrid-openrouter":"Гибридная эволюция: локальный поиск + OpenRouter"}.get(mode, mode)
    def fmt(value: float) -> str:
        """Форматирует дробное число для карточек показателей."""
        return f"{value:.3f}"
    stats = saved.get("statistics", {})
    mcp_events = saved.get("mcp_events", [])
    pareto = pareto_chart(rows, saved, stats)
    profile = best.get("selection_score")
    profile_summary = (f' · оценка профиля {html.escape(str(stats.get("profile", "—")))} {fmt(profile)}'
                       if profile is not None else '')
    screened_attempts = (f'попыток с новыми правилами после train-отсева: <b>{stats["local_evaluations"]}</b>'
                         if "validation_evaluations" in stats else
                         f'локально оценено: <b>{stats.get("local_evaluations", 0)}</b>')
    efficiency = (f'<section class="panel"><h2>Локальный поиск и обращения к LLM</h2>'
                  f'<p>Локальных попыток: <b>{stats.get("local_attempts", 0)}</b>; '
                  f'{screened_attempts}; '
                  f'обращений к LLM: <b>{stats.get("llm_calls", 0)}</b>; '
                  f'время: <b>{stats.get("elapsed_seconds", 0):.1f} с</b>.</p>'
                  + (f'<p>Направленный поиск: {stats.get("guided_attempts", 0)} шагов; '
                     f'{stats.get("guided_train_evaluations", 0)} проверок train.</p>' if stats.get("guided") else "")
                  + (f'<p>Разнообразие: {stats.get("train_behaviors", 0)} поведений train; {stats.get("diverse_steps", 0)} расширенных шагов.</p>' if stats.get("diverse") else "")
                  + f'<p class="muted">Число запросов не является числом токенов. '
                  f'Затраты LLM учитываются отдельно по провайдеру.</p>'
                  f'<p>MCP: <b>{stats.get("mcp_verified", 0)}</b> подтверждений, '
                  f'<b>{stats.get("mcp_summaries", 0)}</b> сводок, '
                  f'<b>{stats.get("mcp_errors", 0)}</b> ошибок.</p></section>') if stats else ""
    usage = read_usage(folder)
    usage_html = (f'<p>HTTP-попыток: {usage["http_attempts"]}; известных токенов: {usage["total_tokens"]}; '
                  f'ответов с usage: {usage["known_usage_responses"]}/{usage["responses"]}. '
                  'Неизвестный расход не подставляется как фактический ноль.</p>' if usage["usage_available"]
                  else '<p>Журнал API-попыток отсутствует (старый запуск).</p>')
    audit = ""
    if mcp_events:
        audit_rows = []
        for event in mcp_events:
            status = "подтверждено" if event.get("ok") else "ошибка"
            delta = event.get("delta_fitness")
            information = (f'Δfitness = {delta:+.3f}'
                           if event.get("ok") and isinstance(delta, (int, float))
                           else event.get("error", "сводка получена"))
            audit_rows.append(
                f'<tr><td>{event.get("generation", "—")}</td>'
                f'<td>{html.escape(str(event.get("tool", "—")))}</td>'
                f'<td>{status}</td><td>{html.escape(str(information))}</td></tr>'
            )
        audit = ('<section class="panel"><h2>MCP: контроль поколений</h2>'
                 '<p class="muted">MCP сверяет метрики лидеров на контрольных точках. '
                 'Δfitness сравнивает скалярный F1 − λ × FPR. При Pareto-отборе он может '
                 'снизиться у нового лидера с более высокой оценкой профиля.</p>'
                 '<div class="scroll"><table style="width:100%;border-spacing:12px 8px;'
                 'text-align:left"><tr><th>Поколение</th><th>Инструмент</th>'
                 '<th>Статус</th><th>Результат</th></tr>'
                 + "".join(audit_rows) + '</table></div></section>')
    # Локальный цикл меняет JSON-правила; старые режимы меняют Python-код через LLM.
    is_rule_mode = mode in ("local", "hybrid", "hybrid-groq", "hybrid-openrouter")
    if is_rule_mode:
        stages = [
            ("01", "Исходное правило", "Признак ≥ 20"),
            ("02", "Мутация", "20 → 18, пример"),
            ("03", "Оценка", "F1 · FPR на CPU"),
            ("04", "Отбор", "Лучшее → в архив"),
            ("05", "Повтор", "Новый родитель"),
        ]
        flow_note = ("Порог — условный пример, не результат этого запуска. "
                     "При застое гибрид может запросить правило у LLM.")
    else:
        stages = [
            ("01", "Seed-код", "Стартовая версия"),
            ("02", "LLM-изменение", "Новый Python-код"),
            ("03", "Проверка", "Код + метрики"),
            ("04", "Отбор", "Лучшее → в архив"),
            ("05", "Повтор", "Новый родитель"),
        ]
        flow_note = "Показаны этапы генерации кода; реальные версии и оценки — ниже."
    flow = ''.join(
        f'<div class="stage"><span>{no}</span><strong>{html.escape(title)}</strong>'
        f'<small>{html.escape(desc)}</small></div>'
        for no, title, desc in stages
    )
    architecture = (
        '<details class="architecture"><summary>Мини-схема архитектуры</summary>'
        '<p>Данные → JSON-правила → CPU-оценщик → отбор и архив ↺</p>'
        '<p class="muted">MCP: после поколения координатор → сервер метрик → контрольная оценка. '
        'Локальный цикл вызывает оценщик напрямую; LLM используется при застое в гибриде.</p>'
        '</details>'
        if is_rule_mode else ''
    )
    records = []
    indexed = {str(row["id"]): row for row in rows}
    for index, row in enumerate(rows):
        source_path = candidate_path(folder, row["path"])
        source = source_path.read_text(encoding="utf-8") if source_path.is_file() else "Код программы отсутствует"
        parent = indexed.get(str(row.get("parent")))
        previous = parent if parent is not None else (rows[index - 1] if index else row)
        old_path = candidate_path(folder, previous["path"])
        old_source = old_path.read_text(encoding="utf-8") if old_path.is_file() else "Код программы отсутствует"
        relationship = "родителем" if parent is not None else "предыдущей оценённой версией"
        records.append(
            f'<details id="record-{html.escape(str(row["id"]))}"><summary>'
            f'<strong>{html.escape(str(row["id"]))} · пок. {row["generation"]}</strong>'
            f'<span>F1 {fmt(row["f1"])} · FPR {fmt(row["fpr"])} · fitness {fmt(row["fitness"])}</span>'
            f'</summary><p class="muted">Сравнение с {relationship} {html.escape(str(previous["id"]))}; '
            f'источник: {html.escape(str(row.get("model") or row.get("kind") or "—"))}</p>'
            '<h3>Изменения программы (+ добавлено, − удалено)</h3>'
            + code_diff(old_source, source, str(previous["id"]), str(row["id"]))
            + '<h3>Полный код</h3><pre><code>' + html.escape(source) + '</code></pre></details>'
        )
    style="""
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;background:#101721;color:#ebf0f7}*{box-sizing:border-box}body{margin:0}main{max-width:1160px;margin:auto;padding:34px 22px 80px}h1{font-size:clamp(27px,4vw,42px);margin:8px 0 8px;letter-spacing:-.6px}h2{font-size:20px;margin:0 0 15px}p{line-height:1.6} .muted{color:#a2b0c3}.eyebrow{color:#4ad5c2;letter-spacing:2px;font-weight:700;font-size:12px}.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:25px 0}.stat,.panel,.stage,details{border:1px solid #314057;border-radius:14px;background:#1a2432}.stat{padding:18px}.stat b{font-size:29px;display:block;margin-top:7px}.stat span{color:#adbbca;font-size:13px}.panel{padding:24px;margin:18px 0}.flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px}.stage{padding:12px;min-height:99px;display:flex;flex-direction:column;gap:7px}.stage span{color:#4ad5c2;font-size:12px;font-weight:800}.stage small{color:#9dadc2;font-size:13px}.scroll{overflow:auto}.panel svg{max-width:100%;height:auto}.bar{background:#253448;border-radius:7px;overflow:hidden;height:16px;margin:9px 0}.bar div{height:100%;background:linear-gradient(90deg,#2c9486,#50d6bf)}details{margin:9px 0;padding:14px 17px;scroll-margin-top:20px}.architecture{padding:10px 12px;margin-top:8px}.architecture p{font-size:12px;margin:8px 0}summary{cursor:pointer;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}summary span{color:#9eafc3;font-variant-numeric:tabular-nums}pre{border:1px solid #314057;border-radius:9px;background:#0e1622;padding:18px;overflow-x:auto;color:#d7e7ff;font-size:13px}a{color:#50d6bf}.foot{font-size:13px;color:#9dadc2;margin-top:35px}.delta{display:block;white-space:pre}.delta-plus{color:#97edcd;background:#163931}.delta-minus{color:#ffb5b7;background:#49232c}.delta-meta{color:#93b7ff;background:#202d48}.code-diff{max-height:380px}.milestones{display:flex;gap:12px;overflow-x:auto;padding:5px 0 16px}.milestone{min-width:140px;display:flex;flex-direction:column;gap:7px;border:1px solid #34516a;border-radius:11px;padding:15px;background:#162737;text-decoration:none}.milestone strong{color:#ebf0f7}.milestone b{font-size:22px;color:#4ad5c2}.milestone small{color:#a4b3c7}h3{font-size:14px;color:#c4d8ef;margin:18px 0 8px}@media(max-width:900px){.flow{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:750px){.stats{grid-template-columns:repeat(2,1fr)}.flow{grid-template-columns:repeat(2,1fr)}}@media(max-width:480px){.stats,.flow{grid-template-columns:1fr}main{padding:22px 14px}}
"""
    body = f'''<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Эволюция сетевого детектора</title><style>{style}</style></head><body><main>
<div class="eyebrow">ЭВОЛЮЦИЯ КОДА</div><h1>Эволюционный анализатор сетевых логов</h1><p class="muted">{html.escape(label)}. Измерения выполнены на {html.escape(data_label)}. Все графики показывают фактические результаты данного запуска.</p>
<div class="stats"><div class="stat"><span>Начальный F1</span><b>{fmt(first['f1'])}</b></div><div class="stat"><span>Максимальный F1</span><b>{fmt(max(item['f1'] for item in rows))}</b></div><div class="stat"><span>FPR лучшего</span><b>{fmt(best['fpr'])}</b></div><div class="stat"><span>Оценено программ</span><b>{len(rows)}</b></div></div>
<section class="panel"><h2>Как эволюционирует правило</h2><div class="flow">{flow}</div><p class="muted">{html.escape(flow_note)}</p>{architecture}</section>
{efficiency}
<section class="panel"><h2>Расход API</h2>{usage_html}<p>Штраф FPR: {float(stats.get("fpr_penalty", 0.2)):.2f}.</p></section>
{audit}
{line_chart(rows,'f1','F1-score по оценкам','#4ad5c2')}
{line_chart(rows,'fitness','Fitness по оценкам','#4ad5c2')}
{pareto}
{line_chart(rows,'fpr','Ложные срабатывания (FPR)','#f3ad62')}
<section class="panel"><h2>Сравнение</h2><p>Начальный F1: <b>{fmt(first['f1'])}</b> → F1 лидера: <b>{fmt(best['f1'])}</b> · скалярный fitness {fmt(best['fitness'])}{profile_summary}</p><div class="bar"><div style="width:{min(100,best['f1']*100):.2f}%"></div></div><p class="muted">Событий отсева до validation: {rejected}. Лидер: {html.escape(best['id'])}, поколение {best['generation']}.</p></section>
{best_timeline(rows)}
{lineage_chart(rows)}
<section class="panel"><h2>Все принятые программы</h2><p class="muted">Открой любую строку: увидишь изменения относительно родителя (если связь известна) или предыдущей оценённой версии, а также полный Python-код.</p>{''.join(records)}</section>
<p class="foot">Исследовательский прототип. Для подтверждения результатов нужен отдельный тест на неиспользованных при отборе программах и данных.</p></main></body></html>'''
    report = folder / "report.html"
    report.write_text(body, encoding="utf-8")
    return report


def main() -> None:
    """Создаёт HTML по указанному каталогу результатов."""
    parser = argparse.ArgumentParser(description="Создать HTML-визуализацию эволюции")
    parser.add_argument("directory", help="Каталог с history.json или history.jsonl")
    args = parser.parse_args()
    print(create_report(Path(args.directory)))


if __name__ == "__main__":
    main()
