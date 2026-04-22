# geometry.py
import cv2
import numpy as np
import math
from config import SAFE_CLASSES, MAX_SLOPE, W_DIST, W_ROUGH, W_SEM, NORM_DIST_DENOMINATOR, BOX_WIDTH, BOX_HEIGHT, ROTATION_ANGLES
from knowledge_graph import think
from semantic_brain import identify_dominant_terrain

def generate_safe_mask(seg_mask):
    """Creates a binary mask (255=Safe, 0=Unsafe)"""
    binary_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for c in SAFE_CLASSES:
        binary_mask[seg_mask == c] = 255
    return binary_mask

def get_rotated_rect_points(center, width, height, angle_deg):
    """Generates the 4 corner points of a rotated bounding box."""
    rect = ((center[0], center[1]), (width, height), angle_deg)
    box = cv2.boxPoints(rect)
    return np.int32(box)

def extract_interior_points(center, width, height, angle_deg, img_shape):
    """Returns grid of points strictly inside the bounding box."""
    # Simplified for speed: generate mask of box, extract coordinates
    mask = np.zeros(img_shape, dtype=np.uint8)
    box = get_rotated_rect_points(center, width, height, angle_deg)
    cv2.fillPoly(mask, [box], 255)
    points = np.column_stack(np.where(mask > 0)) # returns (y, x)
    return points[:, [1, 0]] # Swap to (x, y)

def compute_roughness(depth_map, interior_points):
    """Calculates mean Sobel gradient magnitude for the interior points."""
    # FIX: Temporarily scale the 0-255 depth map down to 0.0-1.0 
    # so the math aligns with the strict 0.15 MAX_SLOPE threshold.
    norm_depth = depth_map.astype(np.float64) / 255.0
    
    sobelx = cv2.Sobel(norm_depth, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(norm_depth, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    ys, xs = interior_points[:, 1], interior_points[:, 0]
    mean_mag = np.mean(magnitude[ys, xs])
    return mean_mag

def calculate_distance(center, target):
    """Normalised Euclidean Distance"""
    dist = math.sqrt((center[0] - target[0])**2 + (center[1] - target[1])**2)
    return dist / NORM_DIST_DENOMINATOR

def search_best_landing_zone(seg_mask, depth_map, target_coord, active_nodes):
    safe_mask = generate_safe_mask(seg_mask)
    h, w = safe_mask.shape
    best_candidate = None
    lowest_cost = float('inf')
    
    terrain_penalties = think(active_nodes)

    # Grid search (step by 20 pixels to optimize search speed)
    for y in range(BOX_HEIGHT//2, h - BOX_HEIGHT//2, 20):
        for x in range(BOX_WIDTH//2, w - BOX_WIDTH//2, 20):
            if safe_mask[y, x] == 0: continue # Skip trivially unsafe centers
            
            for angle in ROTATION_ANGLES:
                interior_points = extract_interior_points((x, y), BOX_WIDTH, BOX_HEIGHT, angle, (h, w))
                
                # Check 1: Are all interior points within image bounds?
                if not (np.all(interior_points[:, 0] >= 0) and np.all(interior_points[:, 0] < w) and
                        np.all(interior_points[:, 1] >= 0) and np.all(interior_points[:, 1] < h)):
                    continue
                
                # Check 2: Are all interior points safe?
                ys, xs = interior_points[:, 1], interior_points[:, 0]
                if np.any(safe_mask[ys, xs] == 0):
                    continue
                
                # Check 3: Roughness / Slope evaluation
                raw_roughness = compute_roughness(depth_map, interior_points)
                if raw_roughness > MAX_SLOPE:
                    continue # Strictly Discarded
                
                norm_roughness = raw_roughness / MAX_SLOPE
                
                # Check 4: Semantic Penalty
                terrain = identify_dominant_terrain(seg_mask, interior_points)
                sem_penalty = terrain_penalties[terrain]
                
                # Integration
                dist_score = calculate_distance((x, y), target_coord)
                
                total_cost = (W_DIST * dist_score) + (W_ROUGH * norm_roughness) + (W_SEM * sem_penalty)
                
                if total_cost < lowest_cost:
                    lowest_cost = total_cost
                    best_candidate = {
                        "center": (x, y), "angle": angle, "terrain": terrain,
                        "cost": total_cost, "dist": dist_score, 
                        "roughness": norm_roughness, "sem_penalty": sem_penalty
                    }

    return best_candidate, safe_mask