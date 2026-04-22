# main.py
import argparse
import cv2
import torch
import torchvision.transforms as T
import numpy as np
from config import DEVICE, RAW_IMG_SIZE, ANN_INPUT_SIZE, BEST_MODEL_PATH, MISSION_CONFIG_PATH
from model import SemanticDroneANN
from semantic_brain import parse_mission_config
from geometry import search_best_landing_zone
from utils import render_dashboard

def run_pipeline(img_path, target_x, target_y):
    print(f"--- Processing {img_path} | Target: ({target_x}, {target_y}) ---")
    target_coord = (target_x, target_y)
    
    # 1. Load Configurations & Data
    active_nodes = parse_mission_config(MISSION_CONFIG_PATH)
    print(f"Active Mission Traits: {active_nodes}")
    
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (RAW_IMG_SIZE[1], RAW_IMG_SIZE[0])) # 800x600

    # 2. ANN Semantic Segmentation
    print("Running Semantic Segmentation...")
    ann_model = SemanticDroneANN().to(DEVICE)
    ann_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
    ann_model.eval()
    
    transform = T.Compose([
        T.ToTensor(), T.Resize(ANN_INPUT_SIZE),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = ann_model(input_tensor)
        seg_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy().astype(np.uint8)
        
    # Upscale back to raw size without losing true scale (NEAREST)
    seg_mask_full = cv2.resize(seg_mask, (RAW_IMG_SIZE[1], RAW_IMG_SIZE[0]), interpolation=cv2.INTER_NEAREST)

    # 3. Monocular Depth Estimation (MiDaS via torch.hub)
    print("Running Depth Estimation (MiDaS)...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(DEVICE)
    midas.eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
    
    input_batch = midas_transforms(raw_img).to(DEVICE)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=RAW_IMG_SIZE, mode="bilinear", align_corners=False
        ).squeeze()
        
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 4. Search and Evaluate
    print("Evaluating placement regions...")
    best_candidate, safe_mask = search_best_landing_zone(seg_mask_full, depth_map, target_coord, active_nodes)

    if best_candidate:
        print(f"Optimal Location Found at: {best_candidate['center']} with Cost {best_candidate['cost']:.4f}")
    else:
        print("WARNING: No safe placement found!")

    # 5. Output rendering
    render_dashboard(raw_img, seg_mask_full, depth_map, safe_mask, best_candidate, target_coord)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone AI Landing Evaluator")
    parser.add_argument("img_path", type=str, help="Path to input image")
    parser.add_argument("target_x", type=int, help="Target pixel X coordinate")
    parser.add_argument("target_y", type=int, help="Target pixel Y coordinate")
    args = parser.parse_args()
    
    run_pipeline(args.img_path, args.target_x, args.target_y)