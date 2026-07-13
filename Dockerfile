# Serves web/backend/ (the FastAPI inverse-design + forward-solve API) on
# Hugging Face Spaces (Docker SDK, which expects port 7860). This image
# deliberately does NOT include jupyter/matplotlib/onnx/dev tooling -- only
# what's needed to serve the API: graydiff's core deps (numpy, scipy, torch)
# plus the `web` extra (fastapi, uvicorn).
FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# CPU-only torch explicitly: the default PyPI torch wheel on Linux pulls in
# several GB of CUDA runtime dependencies that a free-tier CPU Space has no
# use for (no GPU to target). This keeps the image small and the build fast.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch

RUN uv pip install --system numpy scipy fastapi "uvicorn[standard]" python-multipart pydantic

COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY web/__init__.py web/__init__.py
COPY web/backend/ web/backend/
COPY models/checkpoints/surrogate_rollout.pt models/checkpoints/surrogate_rollout.pt
COPY data/phase_diagram_cache/phase_diagram.json data/phase_diagram_cache/phase_diagram.json

RUN uv pip install --system -e . --no-deps

EXPOSE 7860
CMD ["uvicorn", "web.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
