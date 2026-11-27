/* Локальная визуализация поколений: без фреймворков и внешних библиотек. */
'use strict';

const state = {token: '', run: null, selected: null, codeRequest: '', rows: [], lastStatus: ''};
const $ = id => document.getElementById(id);
const merit = row => Number(row.selection_score ?? row.fitness);
const fmt = value => value != null && Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';

/** Показывает ошибку API без вставки HTML в документ. */
function showError(message) {
  $('error').textContent = message || '';
  $('error').style.display = message ? 'block' : 'none';
}

/** Отправляет команду серверу с защитным токеном. */
async function command(path, payload = {}) {
  const response = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'},
                                  body: JSON.stringify({...payload, token: state.token})});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Ошибка запроса');
  return data;
}

/** Запускает выбранный режим эволюции. */
async function startRun() {
  showError('');
  try {
    const number = id => $(id).value === '' ? null : Number($(id).value);
    const data = await command('/api/start', {mode: $('mode').value, dataset: $('dataset').value,
      budget: $('budget').value, profile: $('profile').value,
      generations: number('generations'), offspring: number('offspring'), train_size: number('trainSize'),
      validation_size: number('validationSize'), time_limit: number('timeLimit'), max_fpr: number('maxFpr'),
      numeric_threads: number('numericThreads'),
      min_recall: number('minRecall'), llm_budget: $('mode').value === 'hybrid' ? number('llmBudget') || 0 : 0,
      seed: number('seed') ?? 42});
    state.run = data.id;
    state.lastLeader = null;
    state.selected = null;
    state.codeRequest = '';
    await refresh();
  } catch (error) { showError(error.message); }
}

/** Останавливает эксперимент, не удаляя результаты предыдущих оценок. */
async function stopRun() {
  try { await command('/api/stop'); await refresh(); }
  catch (error) { showError(error.message); }
}

/** Рисует SVG-график кандидатов и рекорда метрики на каждом шаге. */
function drawChart(rows) {
  const svg = $('chart');
  svg.replaceChildren();
  const ns = 'http://www.w3.org/2000/svg';
  const element = (tag, attrs = {}, parent = svg) => {
    const child = document.createElementNS(ns, tag);
    for (const [key,value] of Object.entries(attrs)) child.setAttribute(key, String(value));
    parent.appendChild(child);
    return child;
  };
  const x0 = 52, y0 = 22, width = 798, height = 225;
  const metric = $('metric').value;
  const lower = metric === 'fitness' ? Math.min(-0.2, ...rows.map(row=>Number(row.fitness))) : 0;
  const upper = 1;
  const y = value => y0 + (upper - Math.max(lower, Math.min(upper, value))) / (upper-lower) * height;
  const x = index => x0 + index / Math.max(1,rows.length-1) * width;
  for (let i=0; i<=4; i++) {
    const value = lower + (upper-lower) * (4-i) / 4;
    element('line',{x1:x0,y1:y(value),x2:x0+width,y2:y(value),stroke:'#304258','stroke-dasharray':'4 5'});
    const text = element('text',{x:x0-10,y:y(value)+4,fill:'#9fb0c7','text-anchor':'end','font-size':12});
    text.textContent = value.toFixed(2);
  }
  const xlabel = element('text',{x:850,y:282,fill:'#9fb0c7','text-anchor':'end','font-size':12});
  xlabel.textContent = rows.length ? `${rows.length} оценок` : 'Пока нет оценок';
  if (!rows.length) return;
  const values = rows.map(row => Number(row[metric] || 0));
  const best = []; let leader = rows[0];
  for (const row of rows) {
    if (merit(row)>merit(leader)) leader=row;
    best.push(Number(leader[metric]));
  }
  const plot = (series, color, dash='') => {
    element('polyline',{points:series.map((v,i)=>`${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' '),
                        stroke:color,'stroke-width':3,fill:'none','stroke-linejoin':'round','stroke-dasharray':dash});
  };
  plot(values,'#79a9ff'); plot(best,'#4ee0c0','7 5');
  values.forEach((value,i) => {
    const circle = element('circle',{cx:x(i),cy:y(value),r:state.selected===rows[i].id?6:3.5,
                                      fill:'#79a9ff',stroke:state.selected===rows[i].id?'#fff':'none',
                                      'stroke-width':2,style:'cursor:pointer'});
    const tip = element('title',{},circle);
    tip.textContent = `${rows[i].id} · F1 ${fmt(rows[i].f1)} · fitness ${fmt(rows[i].fitness)} · FPR ${fmt(rows[i].fpr)}`;
    circle.addEventListener('click',()=>selectCandidate(rows[i].id));
  });
}

/** Заполняет горизонтальную ленту принятых программ. */
function drawTimeline(rows) {
  const container = $('timeline');
  container.replaceChildren();
  if (!rows.length) {container.textContent='Оценённые версии появятся здесь.'; return;}
  const best = rows.find(row => row.id === state.winnerId) || rows.reduce((a,b)=>merit(b)>merit(a)?b:a, rows[0]);
  for (const row of rows) {
    const node = document.createElement('button');
    node.className = 'node' + (row.id===state.selected?' active':'') + (row.id===best.id?' best':'');
    const label = document.createElement('b'); label.textContent=`${row.id} · пок. ${row.generation}`;
    const detail = document.createElement('small'); detail.textContent=row.model||row.kind||'код';
    const metric = document.createElement('span'); metric.textContent=`F1 ${fmt(row.f1)}`;
    node.append(label,detail,metric);
    node.addEventListener('click',()=>selectCandidate(row.id));
    container.appendChild(node);
  }
}

/** Выбирает программу и загружает текстовый diff с сервера. */
function selectCandidate(candidate) {
  state.selected = candidate;
  state.codeRequest = '';
  drawTimeline(state.rows);
  drawChart(state.rows);
  void refreshCode();
}

/** Выделяет добавленные и удалённые строки без исполнения кода кандидата. */
function showDiff(lines) {
  const pre = $('diff'); pre.replaceChildren();
  if (!lines.length) {pre.textContent='Эта версия не отличается от выбранной для сравнения.';return;}
  for (const line of lines) {
    const item = document.createElement('span'); item.className = 'diffline';
    if (line.startsWith('+++')||line.startsWith('---')||line.startsWith('@@')) item.classList.add('diffhead');
    else if (line.startsWith('+')) item.classList.add('added');
    else if (line.startsWith('-')) item.classList.add('removed');
    item.textContent = line || ' ';
    pre.appendChild(item);
  }
}

/** Запрашивает исходник выбранной программы и сравнение с родителем или Seed. */
async function refreshCode() {
  if (!state.run || !state.selected) return;
  const key = `${state.run}:${state.selected}:${$('compare').value}`;
  if (state.codeRequest === key) return;
  state.codeRequest = key;
  try {
    const response = await fetch(`/api/code?run=${encodeURIComponent(state.run)}&id=${encodeURIComponent(state.selected)}&compare=${encodeURIComponent($('compare').value)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error||'Не удалось загрузить код');
    if (state.codeRequest!==key) return;
    $('source').textContent=data.after;
    $('diffTitle').textContent=`Изменения относительно ${data.base} · ${data.title}`;
    showDiff(data.diff);
    const row=state.rows.find(item=>item.id===state.selected);
    $('selectedMeta').textContent=row?`Кандидат ${row.id} · поколение ${row.generation} · fitness ${fmt(row.fitness)}`:'';
  } catch (error) {
    $('diff').textContent=`Ошибка: ${error.message}`;
    state.codeRequest='';
  }
}

/** Показывает компромисс F1/FPR и счётчики стратегий. */
function drawPareto(rows, ids, attempts, winner) {
  const svg = $('pareto');
  $('paretoPanel').hidden = !ids.length;
  svg.replaceChildren();
  if (!ids.length) return;
  const ns = 'http://www.w3.org/2000/svg';
  const add = (tag, attrs, content, parent = svg) => {
    const node = document.createElementNS(ns, tag);
    for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, val);
    if (content) node.textContent = content;
    parent.append(node);
    return node;
  };
  add('path', {d:'M40 18 V145 H378', fill:'none', stroke:'currentColor'});
  add('text', {x:9,y:22,fill:'currentColor'}, 'F1');
  add('text', {x:346,y:169,fill:'currentColor'}, 'FPR');
  add('text', {x:26,y:163,fill:'currentColor'}, '0');
  add('text', {x:369,y:143,fill:'currentColor'}, '1');
  for (const row of rows.filter(item=>ids.includes(item.id))) {
    const dot = add('circle', {cx:40+row.fpr*335,cy:145-row.f1*120,r:row.id===winner?6:4,
      fill:row.id===winner?'#4ad5c2':'#79a9ff'});
    add('title', {}, `${row.id} · F1 ${fmt(row.f1)} · FPR ${fmt(row.fpr)}`, dot);
  }
  const labels = {mutation:'Мутация', crossover:'Скрещивание', guided:'Направленный', diversity:'Разнообразие', error:'По ошибкам'};
  $('strategies').textContent = Object.entries(attempts).map(([key, value]) => `${labels[key] || key} ${value}`).join(' · ');
}

/** Обновляет карточки метрик, график и список версий. */
function render(data) {
  const rows=data.evaluations || [];
  state.rows=rows;
  const running=data.status==='running';
  $('start').disabled=running || state.globalRunning;
  for (const input of document.querySelectorAll('.controls input, .controls select')) input.disabled = running;
  $('stop').disabled=!running;
  $('dot').className='dot '+data.status;
  const statuses={idle:'Нет активного эксперимента',running:'Эволюция выполняется',completed:'Эксперимент завершён',failed:'Эксперимент завершился с ошибкой',stopped:'Остановлено',archived:'Архивный запуск · исход завершения неизвестен'};
  $('status').textContent=statuses[data.status]||data.status;
  $('runName').textContent=data.id?`Запуск: ${data.id}${data.mode==='demo'?' · пример без LLM':''}`:'';
  $('startF1').textContent=rows.length?fmt(rows[0].f1):'—';
  const winner=rows.find(row=>row.id===data.winner_id) || (rows.length?rows.reduce((a,b)=>merit(b)>merit(a)?b:a):null);
  state.winnerId=winner?.id;
  $('bestF1').textContent=winner?fmt(winner.f1):'—';
  $('bestFpr').textContent=winner?fmt(winner.fpr):'—';
  $('winnerStats').textContent=winner?`Лидер ${winner.id} · ${winner.model||winner.kind||'—'} · fitness ${fmt(winner.fitness)}${winner.selection_score == null ? '' : ` · профиль ${fmt(winner.selection_score)}`} · max F1 ${fmt(Math.max(...rows.map(row=>Number(row.f1))))}`:'';
  const usage=data.usage;
  $('usageStats').textContent=usage?.usage_available?`HTTP-попыток: ${usage.http_attempts} · известные токены: ${usage.total_tokens} (${usage.known_usage_responses}/${usage.responses} ответов с usage)`:'Токены: нет журнала для старого запуска';
  $('bestRecall').textContent=fmt(winner?.recall);
  $('count').textContent=String(rows.length);
  $('rejected').textContent=`Событий отсева: ${data.rejected||0}`;
  const q = data.quota;
  $('quota').textContent=['local', 'demo'].includes(data.mode) ? 'API-запросы: 0 · полностью локально' : q ? `Квота: ${q.windows.minute.requests}/${q.limits.RPM} в мин · ${q.windows.day.requests}/${q.limits.RPD} в сутки` +
    (q.cooldown_seconds>0 ? ` · ожидание ${Math.ceil(q.cooldown_seconds)} с` : '') : 'Квота: —';
  const st = data.statistics || {};
  $('workStats').textContent=Number.isFinite(st.local_attempts)
    ? `Локальные шаги: ${st.local_attempts}${st.validation_evaluations == null ? '' : ` · шаги с новыми правилами: ${st.local_evaluations}`} · проверки правил на train: ${st.train_evaluations ?? st.guided_train_evaluations ?? '—'} · LLM: ${st.llm_calls} · время: ${Number(st.elapsed_seconds || 0).toFixed(1)} с`
    : '';
  if(st.diverse) $('workStats').textContent += ` · поведений train: ${st.train_behaviors||0} · расширенных шагов: ${st.diverse_steps||0}`;
  $('mcpStats').textContent = Number.isFinite(st.mcp_calls) ? `MCP: ${st.mcp_verified} подтверждено · ${st.mcp_errors} ошибок · вызовов ${st.mcp_calls}` : '';
  $('logs').textContent=(data.logs||[]).join('\n')||'Ожидание новых результатов...';
  $('reportLink').classList.toggle('hide',!data.report_ready);
  if(data.report_ready) $('reportLink').href=`/report?run=${encodeURIComponent(data.id)}`;
  if (!rows.length) {
    // Новый или неудачный запуск не должен показывать код предыдущего эксперимента.
    state.selected = null;
    state.lastLeader = null;
    state.codeRequest = '';
    $('selectedMeta').textContent = 'Пока нет оценённых версий';
    $('diffTitle').textContent = 'Изменения Python-кода';
    $('diff').textContent = 'После первой оценки здесь появятся изменения';
    $('source').textContent = 'После первой оценки здесь появится код детектора';
  }
  if (rows.length && !rows.some(row=>row.id===state.selected)) {
    state.selected=winner.id;
    state.codeRequest='';
  }
  if (rows.length) {
    const best=winner;
    // Выделяем нового лидера только при смене лучшего fitness, если не выбран другой вручную.
    if (!state.lastLeader || state.lastLeader !== best.id && state.selected===state.lastLeader) {
      state.selected=best.id; state.codeRequest='';
    }
    state.lastLeader=best.id;
  }
  drawPareto(rows, data.pareto_ids || [], st.strategy_attempts || {}, winner?.id);
  drawTimeline(rows);
  drawChart(rows);
  void refreshCode();
}

/** Периодически получает новые оценки по мере их записи в JSONL/JSON. */
async function refresh() {
  try {
    const requestedRun=state.run;
    const path='/api/state'+(state.run?`?run=${encodeURIComponent(state.run)}`:'');
    const response=await fetch(path,{cache:'no-store'});
    const data=await response.json();
    if (!response.ok) throw new Error(data.error||'Ошибка состояния');
    if (state.run!==requestedRun || (state.run && data.id && state.run!==data.id)) return;
    if(!state.run && data.id) state.run=data.id;
    render(data);
    if (state.lastStatus!==data.status || state.lastRun!==data.id) {
      state.lastStatus=data.status; state.lastRun=data.id; await refreshRuns();
    }
  } catch (error) { showError('Не удалось обновить панель: '+error.message); }
}

/** Переключает раздел без изменения выполняющегося эксперимента. */
function showTab(name) {
  const isRuns=name==='runs';
  $('experimentView').hidden=isRuns; $('runsView').hidden=!isRuns;
  $('tabExperiment').setAttribute('aria-pressed',String(!isRuns));
  $('tabRuns').setAttribute('aria-pressed',String(isRuns));
  if(isRuns) void refreshRuns();
}

/** Открывает текущий или архивный запуск, сбрасывая выбор кандидата. */
function openRun(id) {
  state.run=id||null; state.selected=null; state.lastLeader=null; state.codeRequest='';
  state.lastStatus=null; showTab('experiment'); void refresh();
}

/** Рисует фильтруемую историю; неизвестные значения не заменяются нулём. */
function drawArchive() {
  const table=$('runComparison'); table.replaceChildren();
  const text=$('runFilter').value.toLowerCase(), dataset=$('datasetFilter').value;
  const rows=(state.archive||[]).filter(row=>(!dataset||row.dataset===dataset)&&
    `${row.id} ${row.mode} ${row.status}`.toLowerCase().includes(text));
  const headers=['Запуск','Статус','Данные','Seed','Цель','F1','FPR','Recall','Попытки','Train','HTTP','Токены','Время, с','Источник',''];
  const head=document.createElement('thead'), tr=document.createElement('tr');
  for(const label of headers){const th=document.createElement('th');th.textContent=label;tr.append(th);} head.append(tr); table.append(head);
  const body=document.createElement('tbody'); table.append(body);
  for(const row of rows){
    const tr=document.createElement('tr');
    const usage=row.usage||{};
    const tokens=usage.usage_available?`${usage.total_tokens} (${usage.known_usage_responses}/${usage.responses})`:'—';
    for(const value of [row.id,row.status,row.dataset,row.seed,row.profile,fmt(row.f1),fmt(row.fpr),fmt(row.recall),row.local_attempts,row.train_evaluations,usage.usage_available?usage.http_attempts:'—',tokens,row.seconds,row.source]){
      const td=document.createElement('td');td.textContent=String(value??'—');td.title=td.textContent;tr.append(td);
    }
    const actions=document.createElement('td');actions.className='actions';
    const open=document.createElement('button');open.className='btn ghost';open.textContent='Открыть';open.disabled=state.globalRunning&&row.id!==state.activeRun;open.addEventListener('click',()=>openRun(row.id));actions.append(open);
    if(row.report_ready){const a=document.createElement('a');a.textContent='Отчёт';a.href='/report?run='+encodeURIComponent(row.id);a.target='_blank';a.rel='noopener';actions.append(a);}
    tr.append(actions);body.append(tr);
  }
  $('archiveCount').textContent=`Запусков: ${rows.length} из ${(state.archive||[]).length}`;
}

/** Получает архив и состояние активного процесса. */
async function refreshRuns() {
  try {
    const response=await fetch('/api/runs',{cache:'no-store'});
    if(!response.ok) throw new Error('Ошибка архива');
    const data=await response.json();
    state.globalRunning=data.running; state.activeRun=data.active; state.archive=data.runs;
    $('latestRun').classList.toggle('hide',!state.activeRun||state.run===state.activeRun);
    $('start').disabled=state.globalRunning;
    drawArchive();
  } catch(error) { $('archiveCount').textContent=error.message; }
}

/** Запрашивает настройки интерфейса и открывает демонстрационную историю. */
async function init() {
  try {
    const response=await fetch('/api/config');
    const config=await response.json();
    state.token=config.token;
    state.run=config.initial_run;
    state.budgets = config.budgets || {};
    $('dataset').querySelector('option[value="unsw"]').disabled = !config.unsw_available;
    if (!config.unsw_available) $('dataset').value = 'synthetic';
    $('dataset').title = config.unsw_available ? '' : 'Сначала выполни python -m traffic_data.download_unsw';
    await refresh();
  } catch (error) {showError('Нет связи с локальным сервером: '+error.message);}
  $('tabExperiment').addEventListener('click',()=>showTab('experiment'));
  $('tabRuns').addEventListener('click',()=>showTab('runs'));
  $('latestRun').addEventListener('click',()=>openRun(state.activeRun));
  $('refreshArchive').addEventListener('click',refreshRuns);
  for(const id of ['runFilter','datasetFilter']) $(id).addEventListener('input',drawArchive);
  $('start').addEventListener('click',startRun);
  $('stop').addEventListener('click',stopRun);
  const updateDefaults = () => {
    $('llmSettings').hidden = $('mode').value !== 'hybrid';
    $('dataSettings').hidden = $('dataset').value !== 'unsw';
    const budget = state.budgets?.[$('budget').value] || [];
    ['generations','offspring','trainSize','timeLimit'].forEach((id, index) => {
      $(id).placeholder = budget[index] ?? '';
    });
  };
  for (const id of ['mode','dataset','budget']) $(id).addEventListener('change', updateDefaults);
  updateDefaults();
  $('metric').addEventListener('change',()=>drawChart(state.rows));
  $('compare').addEventListener('change',()=>{state.codeRequest='';void refreshCode();});

  setInterval(refresh,1500);
}
void init();
