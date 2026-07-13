// Points the frontend at the backend API. Local dev default below works
// against `uv run uvicorn web.backend.main:app --port 8001` on the same
// machine. For a deployed frontend (e.g. GitHub Pages), change this to your
// deployed backend's URL, e.g. a Hugging Face Space:
//   window.GRAYDIFF_BACKEND_URL = "https://<your-username>-<space-name>.hf.space";
window.GRAYDIFF_BACKEND_URL = "http://127.0.0.1:8001";
