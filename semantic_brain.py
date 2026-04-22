# semantic_brain.py
import json
import numpy as np
from config import TERRAIN_MAPPING

def parse_mission_config(filepath):
    """Parses mission_config.json to extract active package traits."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    package = data.get("package", {})
    traits = ["heavy", "fragile", "valuable", "biohazard"]
    
    # Extract only boolean true traits and TitleCase them for the graph
    active_nodes = [t.capitalize() for t in traits if package.get(t, False)]
    return active_nodes

def identify_dominant_terrain(segmentation_mask, interior_points):
    """
    Samples interior points, finds most frequent class, maps to terrain.
    """
    # Extract pixel values from the mask at the interior coordinates
    ys, xs = interior_points[:, 1], interior_points[:, 0]
    sampled_classes = segmentation_mask[ys, xs]
    
    # Find dominant class ID
    counts = np.bincount(sampled_classes.flatten())
    dominant_class = counts.argmax()
    
    return TERRAIN_MAPPING.get(dominant_class, "Pavement") # Default to Pavement if unknown