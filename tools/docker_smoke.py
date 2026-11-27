"""Сквозная проверка панели через HTTP без LLM, с реальным MCP в Compose."""
import argparse
import json
import time
import urllib.request


def main() -> None:
    """Создаёт автономный запуск, ждёт отчёт и проверяет два MCP-подтверждения."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    args = parser.parse_args()
    base = args.url.rstrip('/')

    def request(path, data=None):
        """Выполняет локальный HTTP-запрос с исходным Origin."""
        payload = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(base + path, data=payload,
                                     headers={'Content-Type': 'application/json', 'Origin': base})
        with urllib.request.urlopen(req, timeout=10) as reply:
            return reply.read()

    config = json.loads(request('/api/config'))
    run = json.loads(request('/api/start', {'token': config['token'], 'mode': 'local',
        'dataset': 'synthetic', 'generations': 2, 'offspring': 5, 'llm_budget': 0,
        'seed': 42}))['id']
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        state = json.loads(request('/api/state?run=' + run))
        if state['status'] != 'running':
            break
        time.sleep(1)
    else:
        raise RuntimeError('Smoke timeout: ' + run)
    stats = state['statistics']
    if (state['status'] != 'completed' or not state['report_ready'] or stats['mcp_verified'] != 2
            or stats['mcp_errors'] != 0 or stats['llm_calls'] != 0 or not stats['strategy_attempts']):
        raise RuntimeError(f'Smoke failed: {state}')
    if b'<html' not in request('/report?run=' + run).lower():
        raise RuntimeError('Report not available')
    print(json.dumps({'status': 'ok', 'run': run, 'mcp_verified': stats['mcp_verified'],
                      'llm_calls': stats['llm_calls'], 'report': True}, ensure_ascii=False))


if __name__ == '__main__':
    main()
