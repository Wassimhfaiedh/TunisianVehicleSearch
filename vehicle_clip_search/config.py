import os
from dotenv import load_dotenv

load_dotenv()

PLATE_MODEL_PATH = os.environ.get("PLATE_MODEL_PATH", "license_plate_detector.pt")
VEHICLE_MODEL_PATH = os.environ.get("VEHICLE_MODEL_PATH", "yolov8s.pt")

DEFAULT_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
CROP_MARGIN = 6

PLATE_DETECT_CONF = 0.3
PLATE_VLM_CONF_THRESHOLD = 0.6
PENDING_WINDOW_FRAMES = 25

CAPTURES_DIR = os.environ.get("CAPTURES_DIR", "captures")
DB_PATH = os.environ.get("DB_PATH", "vehicle_search_chroma")

NVIDIA_MODEL = "nvidia/nemotron-nano-12b-v2-vl"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
