# config.py
import torch

# --- Device Selection ---

# --- Device Selection (Updated for Mac/Apple Silicon Support) ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda") # For Nvidia GPUs
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # For Mac (Apple Silicon / Metal)
else:
    DEVICE = torch.device("cpu")  # Fallback

print(f"Using device: {DEVICE}")

# --- Image Dimensions ---
RAW_IMG_SIZE = (600, 800)  # Height, Width
ANN_INPUT_SIZE = (256, 256) # Fast inference size

# --- Semantic Segmentation Classes ---
NUM_CLASSES = 23 # 0 to 22
SAFE_CLASSES = [1, 3, 4] # IDs considered safe for landing
TERRAIN_MAPPING = {1: "Pavement", 3: "Dirt", 4: "Grass"}

# --- Cost Function Weights ---
W_DIST = 0.4
W_ROUGH = 0.2
W_SEM = 0.4
MAX_SLOPE = 0.15 # Sobel magnitude threshold
NORM_DIST_DENOMINATOR = 1000.0

# --- Drone / Box footprint ---
BOX_WIDTH = 40
BOX_HEIGHT = 40
ROTATION_ANGLES = [0, 45, 90, 135] # Degrees to test during grid search

# --- Paths ---
BEST_MODEL_PATH = "best_model.pth"
MISSION_CONFIG_PATH = "mission_config.json"