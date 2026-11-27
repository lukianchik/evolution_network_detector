FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements-mcp.txt \
    && groupadd -g 10001 evo && useradd -u 10001 -g evo -m evo \
    && mkdir -p /app/data /app/results /app/.state /app/metrics-data \
    && chown -R evo:evo /app
COPY --chown=evo:evo app.py pyproject.toml ./
COPY --chown=evo:evo artifacts artifacts
COPY --chown=evo:evo detectors detectors
COPY --chown=evo:evo evaluation evaluation
COPY --chown=evo:evo integrations integrations
COPY --chown=evo:evo search search
COPY --chown=evo:evo seeds seeds
COPY --chown=evo:evo tools tools
COPY --chown=evo:evo traffic_data traffic_data
COPY --chown=evo:evo ui ui
COPY --chown=evo:evo tests tests
COPY --chown=evo:evo examples examples
COPY --chown=evo:evo deploy deploy
FROM base AS metrics
USER evo
CMD ["python", "-m", "integrations.mcp_metrics", "--http"]
FROM base AS worker
ARG INSTALL_SHINKA=0
RUN pip install --no-cache-dir 'groq>=0.26,<3' ruff \
    && if [ "$INSTALL_SHINKA" = "1" ]; then pip install --no-cache-dir -r requirements-shinka.txt; fi
USER evo
CMD ["python", "app.py"]
