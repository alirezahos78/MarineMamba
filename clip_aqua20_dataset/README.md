# AQUA20 CLIP Feature Dataset

This folder documents the generated AQUA20 CLIP-feature dataset. The generated
feature files are written under `data/aqua20_clip_features/` by default.

Build the feature dataset with:

```bash
python3 scripts/build_aqua20_clip_features.py
```

Each split produces:

- `<split>_features.pt`: CLIP image features, integer labels, image paths, and
  class names.
- `<split>_metadata.jsonl`: one row per image with `index`, `path`, `label`,
  and `class_name`.

The default CLIP backbone is OpenCLIP `ViT-B-32` with `openai` weights.
