# Mark-1 AI Orchestrator - Production Dockerfile
# Multi-stage build for optimal security and performance

# =====================================
# Stage 1: Builder
# =====================================
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add labels for metadata
LABEL maintainer="Mark-1 AI Team" \
      version="${VERSION}" \
      description="Mark-1 AI Orchestrator - Production Build" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mark1 && useradd -r -g mark1 mark1

# Set work directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the application
RUN pip install --no-cache-dir -e .

# =====================================
# Stage 2: Production Runtime
# =====================================
FROM python:3.11-slim as production

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add labels for metadata
LABEL maintainer="Mark-1 AI Team" \
      version="${VERSION}" \
      description="Mark-1 AI Orchestrator - Production Runtime" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.local/bin:$PATH" \
    MARK1_ENV=production \
    MARK1_LOG_LEVEL=INFO \
    MARK1_HOST=0.0.0.0 \
    MARK1_PORT=8000

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user with specific UID/GID
RUN groupadd -r -g 1001 mark1 && useradd -r -u 1001 -g mark1 mark1

# Create necessary directories
RUN mkdir -p /app /app/logs /app/data /app/models /app/configs && \
    chown -R mark1:mark1 /app

# Set work directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --from=builder --chown=mark1:mark1 /app .

# Create configuration files
RUN echo '{"log_level": "INFO", "environment": "production", "debug": false}' > /app/configs/production.json

# Switch to non-root user
USER mark1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${MARK1_PORT}/health || exit 1

# Expose port
EXPOSE ${MARK1_PORT}

# Add entrypoint script
COPY --chown=mark1:mark1 docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["mark1", "orchestrator", "start", "--host", "0.0.0.0", "--port", "8000"] 