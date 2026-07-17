# Tunisian Vehicle Search Using VLMs and CLIP

This project uses object detection (YOLO) and a vision-language model (VLM) to detect
vehicles and read their license plates exactly as printed — including Tunisian-style
plates (digits + "TN" + digits). Every detected vehicle is also embedded with CLIP, so
you can search your logged vehicles using natural language ("silver peugeot") or by
uploading a photo of a car or a license plate.

## Structure

```
app.py                          # Gradio UI (process video + semantic search)
vehicle_clip_search/
  config.py                     # settings, loaded from .env
  pipeline.py                   # detection + tracking + VLM + storage
  clip_embedder.py               # open_clip image/text embeddings
  vector_store.py                # ChromaDB read/write
requirements.txt
.env.example
```

## Setup

```bash
git clone <your-repo-url>
cd vehicle-clip-search
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place your model weights (`yolov8s.pt`, `license_plate_detector.pt`) in the project
root, or point `VEHICLE_MODEL_PATH` / `PLATE_MODEL_PATH` in `.env` to their location.

## Run

```bash
python app.py
```

Opens at `http://127.0.0.1:7860`.

1. **Process Video tab** — upload a video, click two points on the frame to set the
   crossing line, enter your Nemotron API key, click Process.
2. **Semantic Search tab** — search by text ("silver peugeot") or by uploading a
   car/plate photo.

## Notes

- Get an NVIDIA API key at https://build.nvidia.com.
- `captures/` and `vehicle_search_chroma/` are created at runtime and are gitignored.
- Never commit `.env` or model weights with embedded keys.
