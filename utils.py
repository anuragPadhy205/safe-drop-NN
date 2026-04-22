# utils.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def render_dashboard(orig_img, seg_map, depth_map, safe_mask, best_result, target_coord, output_path="output_analysis.jpg"):
    """Generates the 6-panel requirement."""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Original Image
    axs[0, 0].imshow(orig_img)
    axs[0, 0].set_title("Original Image")
    
    # 2. Segmentation
    axs[0, 1].imshow(seg_map, cmap='tab20')
    axs[0, 1].set_title("Semantic Segmentation")
    
    # 3. Depth
    axs[0, 2].imshow(depth_map, cmap='viridis')
    axs[0, 2].set_title("Depth Map")
    
    # 4. Safe Mask
    axs[1, 0].imshow(safe_mask, cmap='gray')
    axs[1, 0].set_title("Safe Zone Mask")
    
    # 5. Best Placement Overlay
    place_img = orig_img.copy()
    cv2.drawMarker(place_img, target_coord, (255, 0, 0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=2)
    if best_result:
        from geometry import get_rotated_rect_points
        box = get_rotated_rect_points(best_result['center'], 40, 40, best_result['angle'])
        cv2.drawContours(place_img, [box], 0, (0, 255, 0), 2)
        text = f"Cost: {best_result['cost']:.4f}"
        cv2.putText(place_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    axs[1, 1].imshow(place_img)
    axs[1, 1].set_title("Best Placement")
    
    # 6. Cost Breakdown
    axs[1, 2].axis('off')
    if best_result:
        stats = (
            f"BEST PLACEMENT (SEMANTIC REASONING)\n\n"
            f"Position: {best_result['center']}\n"
            f"Angle: {best_result['angle']}°\n"
            f"Terrain: {best_result['terrain']}\n"
            f"Distance Cost: {best_result['dist']:.4f}\n"
            f"Roughness Cost: {best_result['roughness']:.4f}\n"
            f"Semantic Penalty: {best_result['sem_penalty']:.4f}\n\n"
            f"TOTAL COST: {best_result['cost']:.4f}"
        )
        axs[1, 2].text(0.1, 0.5, stats, fontsize=12, verticalalignment='center', bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved dashboard to {output_path}")