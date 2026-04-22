***

# AI Autonomous Landing Zone Analysis Model

This project implements an AI-driven, autonomous drone landing zone selection system.  It utilizes an Artificial Neural Network (ANN) for semantic segmentation, monocular depth estimation, and a Spreading Activation Knowledge Graph to dynamically identify the safest and most optimal landing coordinates for a package drop.

## System Architecture

 The pipeline strictly adheres to a modular architecture and executes in a sequential inference pipeline: **Image Segmentation $\rightarrow$ Depth Estimation $\rightarrow$ Candidate Region Generation $\rightarrow$ Semantic Graph Evaluation $\rightarrow$ Cost Ranking**.

### Core Components:
* **Semantic Segmentation (U-Net):** A custom ANN featuring exactly 4 encoder layers and 4 decoder layers.  Trained on the TU Graz Semantic Drone Dataset to predict 23 per-pixel class labels from a $256 \times 256$ RGB image.
*  **Depth Estimation & Geometry:** Utilizes Intel's MiDaS model (`MiDaS_small` via `torch.hub`) to generate a relative depth map.  Terrain roughness is calculated using a mean Sobel gradient magnitude.
*  **Geometric Safety Filter:** Any candidate bounding box whose interior points exceed a `MAX_SLOPE` threshold of 0.15 is strictly discarded.  The system searches for flat patches entirely within geometrically safe terrain classes (Pavement, Grass, Dirt).
*  **Semantic Reasoning Engine:** A 3-layer Spreading Activation Knowledge Graph parses a `mission_config.json` payload payload file.  It computes terrain penalty scores via edge-weight accumulation (Source Nodes $\rightarrow$ Property Nodes $\rightarrow$ Terrain Nodes) with absolutely no `if/else` logic permitted.

### The Cost Optimization Function
 Candidate regions are ranked using a normalized scalar cost function.  The lowest cost determines the optimal landing spot. 

 $Cost = 0.4 \times distance + 0.2 \times roughness + 0.4 \times semantic\_penalty$ 

---

## Repository Structure

 All source code is contained within the `src/` directory.

*  `config.py`: Centralized configuration for all constants, including $800 \times 600$ image sizes, 22-class semantic labels, cost function weights, and device selection.
*  `dataset.py`: Implements the `GrazDataset` class to load images, translate RGB masks to integer class IDs using `class_dict.csv`, and apply normalization transformations.
*  `model.py`: Defines the core U-Net CNN architecture.
*  `train.py`: Orchestrates model training using Cross-Entropy loss and the Adam optimizer, saving the optimal weights to `best_model.pth`.
*  `geometry.py`: Generates binary safe-zone masks, executes a rotational grid search for the footprint, performs Sobel roughness validation, and calculates the final normalized cost score.
*  `knowledge_graph.py`: Implements the `think()` function for the 3-layer Spreading Activation graph, managing `SOURCE_TO_PROPERTY_EDGES` and `PROPERTY_TO_TERRAIN_EDGES` with capped penalty outputs.
*  `semantic_brain.py`: Parses active mission traits from `mission_config.json` and samples interior points to classify the dominant terrain type for graph penalty lookup.
*  `utils.py`: Contains helper routines for semantic color mapping and rendering the final 6-panel output dashboard.
*  `main.py`: The entry point and orchestration layer for the full inference pipeline.

---

## Installation & Setup

 This project uses `conda` for environment management and natively supports hardware acceleration across CUDA (Nvidia), MPS (Apple Silicon Mac), and CPU architectures.

1.  **Create and activate the environment:**
    ```bash
    conda create -n ai_drone_env python=3.10
    conda activate ai_drone_env
    ```
2.  **Install PyTorch:**
    * *For Mac (Apple Silicon / MPS):* `conda install pytorch torchvision -c pytorch`
    * *For Windows/Linux (CUDA):* `conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia`
3.  **Install required dependencies:**
    ```bash
    conda install -c conda-forge matplotlib numpy timm
    ```
    *Note for macOS users: To avoid `Protobuf` C++ linking errors with PyTorch, install OpenCV via pip rather than conda:*
    ```bash
    pip install opencv-python
    ```

---

## Usage

 Ensure the TU Graz Semantic Drone Dataset is downloaded and properly referenced in `train.py`.

### Phase 1: Train the Model
Train the U-Net model on the dataset to generate `best_model.pth`.
```bash
python3 src/train.py
```

### Phase 2: Run Inference
Test the pipeline on a single image. The script requires the image path and the $(x, y)$ coordinates of the target destination. 
```bash
python3 src/main.py path/to/test_image.jpg 400 300
```
*Outputs:*
*  `output.jpg`: The original image overlaid with the optimal green bounding box and target coordinate.
*  `output_analysis.jpg`: A 6-panel dashboard displaying the Original Image, Semantic Segmentation map, Depth Map, Safe Zone Mask, Best Placement, and a detailed Cost Breakdown.
