FROM python:3.12-slim-bookworm

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Install system build dependencies required for eFEL compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir uv

# Copy dependency definitions first (better layer caching)
COPY pyproject.toml uv.lock ./
RUN uv pip install --system .

COPY app.py .
COPY modules/ ./modules/
COPY assets/ ./assets/

EXPOSE 8000

CMD ["shiny", "run", "--host", "0.0.0.0", "app.py"]
