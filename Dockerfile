# Multi-stage build using uv. The final image contains only the runtime venv
# and the application source — no build tooling.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached layer), then the project.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


FROM python:3.12-slim-bookworm AS runtime

# Non-root runtime user.
RUN useradd --create-home --uid 1000 watari
WORKDIR /app

COPY --from=builder --chown=watari:watari /app /app
ENV PATH="/app/.venv/bin:$PATH"

USER watari
# Container listens on all interfaces *inside* the container; publish it
# deliberately with `-p 127.0.0.1:8000:8000` to keep it host-local.
ENV WATARI_HOST=0.0.0.0
EXPOSE 8000

ENTRYPOINT ["watari"]
CMD ["serve"]
