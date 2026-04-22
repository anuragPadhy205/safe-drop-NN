# knowledge_graph.py

# Layer 1 -> Layer 2 Mapping
SOURCE_TO_PROPERTY_EDGES = {
    "Fragile": {"Soft": 0.0, "Hard": 0.8, "Unstable": 0.5},
    "Valuable": {"Visible": 0.7, "Dirty": 0.4, "Wet": 0.3},
    "Biohazard": {"Contaminated": 0.9, "Wet": 0.6, "Soft": 0.3},
    "Heavy": {"Soft": 0.8, "Unstable": 0.9, "Hard": 0.0}
}

# Layer 2 -> Layer 3 Mapping
PROPERTY_TO_TERRAIN_EDGES = {
    "Hard": {"Pavement": 0.9, "Dirt": 0.4, "Grass": 0.1},
    "Soft": {"Pavement": 0.0, "Dirt": 0.5, "Grass": 0.8},
    "Wet": {"Pavement": 0.2, "Dirt": 0.9, "Grass": 0.6},
    "Slippery": {"Pavement": 0.8, "Dirt": 0.6, "Grass": 0.3},
    "Dirty": {"Pavement": 0.1, "Dirt": 0.9, "Grass": 0.7},
    "Visible": {"Pavement": 0.8, "Dirt": 0.4, "Grass": 0.2},
    "Contaminated": {"Pavement": 0.5, "Dirt": 0.9, "Grass": 0.8},
    "Unstable": {"Pavement": 0.1, "Dirt": 0.7, "Grass": 0.8}
}

def think(active_source_nodes):
    """
    Computes a capped penalty score for each terrain via accumulation.
    Returns: Dict[terrain_string, float_penalty]
    """
    property_scores = {p: 0.0 for p in ["Hard", "Wet", "Slippery", "Dirty", "Visible", "Contaminated", "Soft", "Unstable"]}
    terrain_scores = {"Pavement": 0.0, "Grass": 0.0, "Dirt": 0.0}

    # Layer 1 -> Layer 2 Activation
    for source in active_source_nodes:
        edges = SOURCE_TO_PROPERTY_EDGES.get(source, {})
        for prop, weight in edges.items():
            property_scores[prop] += weight

    # Layer 2 -> Layer 3 Activation
    for prop, activation in property_scores.items():
        edges = PROPERTY_TO_TERRAIN_EDGES.get(prop, {})
        for terrain, weight in edges.items():
            # Multiply accumulated property activation by edge weight
            terrain_scores[terrain] += (activation * weight)

    # Normalize/Cap at 1.0 (without using if/else)
    for t in terrain_scores:
        terrain_scores[t] = min(1.0, terrain_scores[t])

    return terrain_scores