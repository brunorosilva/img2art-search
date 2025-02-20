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

COPY . $HOME/app
WORKDIR $HOME/app

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH    

RUN pip install "poetry==$POETRY_VERSION" && \
    poetry export --without-hashes --format requirements.txt --output $HOME/requirements.txt && \
    python3 -m pip wheel --no-cache-dir --no-deps -w $HOME/app/wheels -r $HOME/requirements.txt

RUN python3 -m pip install --no-cache $HOME/app/wheels/*

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

ENTRYPOINT ["poetry", "run", "python3", "main.py", "interface"]