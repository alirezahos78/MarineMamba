import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT    = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results.json")

# Raw AQUA20 images (created by scripts/download_aqua20.py, or set AQUA20_ROOT env var)
AQUA20_ROOT = os.environ.get("AQUA20_ROOT", os.path.join(DATA_ROOT, "aqua20"))

# Spatial CLIP feature caches (built by scripts/build_clip_features.py)
B16_FEATURES_ROOT = os.path.join(DATA_ROOT, "aqua20_clip_vit_b16_spatial_grid_aug")
B32_FEATURES_ROOT = os.path.join(DATA_ROOT, "aqua20_clip_vit_b32_spatial_grid_aug")

# CLS pooled features (both B/16 and B/32, built by scripts/build_clip_features.py)
CLS_FEATURES_PATH = os.path.join(DATA_ROOT, "dual_clip_pooled_features.pt")

# ── Sea Animals 23 ───────────────────────────────────────────────────────────
SEA23_ROOT     = os.environ.get("SEA23_ROOT", os.path.join(DATA_ROOT, "sea23"))
SEA23_B16_ROOT = os.path.join(DATA_ROOT, "sea23_clip_vit_b16_spatial_grid_aug")
SEA23_B32_ROOT = os.path.join(DATA_ROOT, "sea23_clip_vit_b32_spatial_grid_aug")
SEA23_CLS_PATH = os.path.join(DATA_ROOT, "sea23_dual_clip_pooled_features.pt")

# ── Fish4Knowledge 23 ─────────────────────────────────────────────────────────
FISH4K_ROOT     = os.environ.get("FISH4K_ROOT", os.path.join(DATA_ROOT, "fish4k"))
FISH4K_B16_ROOT = os.path.join(DATA_ROOT, "fish4k_clip_vit_b16_spatial_grid_aug")
FISH4K_B32_ROOT = os.path.join(DATA_ROOT, "fish4k_clip_vit_b32_spatial_grid_aug")
FISH4K_CLS_PATH = os.path.join(DATA_ROOT, "fish4k_dual_clip_pooled_features.pt")
