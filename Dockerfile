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

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY modules/ ./modules/
COPY assets/ ./assets/

# Expose the port Shiny runs on
EXPOSE 8000

# Run the Shiny application
CMD ["shiny", "run", "--host", "0.0.0.0", "app.py"]
