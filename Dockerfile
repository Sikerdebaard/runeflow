# runeflow — General-purpose container
# Provides all CLI commands: update-data, train, inference, export-tariffs
#
# Build:
#   docker build -t runeflow .
#
# Run (tariff export):
#   docker run --rm \
#     -e ENTSOE=<key> -e ZONE=NL \
#     -v $(pwd)/outputs:/outputs \
#     runeflow export-tariffs --zone NL

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=30 \
    TZ=Europe/Amsterdam

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cron \
    tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /app

# Install package (src-layout, hatchling build backend)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Runtime directories
RUN mkdir -p /app/.cache/runeflow /var/log/runeflow /outputs

# Entrypoint
COPY docker/entrypoint-export.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Non-root user
RUN useradd -m -s /bin/bash runeflow && \
    chown -R runeflow:runeflow /app /var/log/runeflow /outputs

# Required env: ENTSOE
# Optional env: ZONE (default: NL), PRICE_PROVIDER (default: wholesale),
#               OUTPUT_FILE, CHECK_INTERVAL, LOG_LEVEL

VOLUME ["/outputs", "/app/.cache/runeflow"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["cron", "-f"]
