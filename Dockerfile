FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VERSION=1.8.3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
WORKDIR /app

RUN pip install "poetry==$POETRY_VERSION" && \
    poetry export --without-hashes --format requirements.txt --output requirements.txt && \
    python3 -m pip wheel --no-cache-dir --no-deps -w /app/wheels -r requirements.txt

RUN python3 -m pip install --no-cache /app/wheels/*

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["poetry", "run", "python3", "main.py", "interface"]