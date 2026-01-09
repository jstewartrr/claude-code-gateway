# Claude Code Gateway - Dockerfile
# Response-limited MCP Gateway for Claude Code

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY gateway.py .

# Environment variables with defaults
ENV MAX_RESPONSE_BYTES=51200
ENV MAX_ROWS=100
ENV MAX_EMAIL_COUNT=25
ENV MAX_TASK_COUNT=50
ENV MAX_FILE_LIST=100
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 1

# Run with SSE transport for remote access
CMD ["python", "gateway.py", "sse"]
