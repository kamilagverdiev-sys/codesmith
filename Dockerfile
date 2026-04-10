# Codesmith API server image.
#
# Build:
#   docker build -t codesmith:latest .
#
# Run:
#   docker run -p 8111:8111 \
#     -v $(pwd)/config.yaml:/app/config.yaml:ro \
#     -v ~/.codesmith:/data \
#     --env-file .env \
#     codesmith:latest
#
# The container needs Docker socket access if you want sandbox execution:
#   -v /var/run/docker.sock:/var/run/docker.sock

FROM python:3.11-slim AS base

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[web]" 2>/dev/null || true

# Copy source
COPY src/ src/
COPY config.example.yaml ./

# Install the package itself
RUN pip install --no-cache-dir -e ".[web]"

# Default config paths
ENV CODESMITH_CONFIG=/app/config.yaml

EXPOSE 8111

CMD ["python", "-m", "uvicorn", "codesmith.api.main:app", \
     "--host", "0.0.0.0", "--port", "8111"]
