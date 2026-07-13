# Running the web app locally

Two independent local servers — no build step for either.

## 1. Backend (needed for "Solve for physics")

```bash
uv run uvicorn web.backend.main:app --reload --port 8001
```

Loads `models/checkpoints/surrogate_rollout.pt` once at startup. Requires notebooks 00 and 03
to have been run at least once (for `data/phase_diagram_cache/phase_diagram.json` and the
checkpoint respectively) — both are already committed to the repo, so this works out of the box.

## 2. Frontend (forward playground works standalone, without the backend)

```bash
cd web/frontend
python3 -m http.server 8000
```

Then open http://127.0.0.1:8000. The forward playground runs the ONNX model entirely in the
browser and needs no backend. "Solve for physics" calls the backend at `http://127.0.0.1:8001`
by default — set `window.GRAYDIFF_BACKEND_URL` before `app.js` loads to point elsewhere.
