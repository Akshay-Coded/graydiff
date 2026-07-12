# Differentiable Gray-Scott Inverse Design

A neural surrogate for the Gray-Scott reaction-diffusion PDE, trained to be **differentiable**
in its physics parameters (F, k) — so that, unlike the classical numerical solver, it can be
run *backwards*: given a target pattern, gradient descent through the frozen network recovers
the (F, k) that produce it. Every recovered parameter is verified against the real NumPy solver.

See `Inverse_Design_Differentiable_Surrogate_Spec.docx` for the full research spec this project
implements, and `notebooks/` for the phase-by-phase build with embedded EDA and explanation.

## Setup

```bash
uv sync --extra notebook --extra export --extra web --extra dev
uv run jupyter lab
```

## Project layout

- `src/graydiff/` — installable package: solver, data generation, model, training, losses,
  inverse design, ONNX export. Single source of truth imported by notebooks, tests, and the backend.
- `notebooks/` — one notebook per build phase, EDA and explanation interleaved with code.
- `tests/` — pytest suite for the package (`uv run pytest -m "not slow"` for the fast loop).
- `web/frontend/` — static site (draw-a-target canvas, forward playground, phase-diagram view).
- `web/backend/` — FastAPI service for the inverse-design solve (needs PyTorch backprop, so it
  can't run purely client-side like the ONNX forward playground).

## Build order

Phases 0-4 build and validate the forward surrogate (the foundation). Phase 4's solver-vs-surrogate
phase-diagram match is a hard gate — inverse design (Phase 5) does not proceed without it. Phases
6-7 export and deploy. See the plan for the full milestone checklist.
