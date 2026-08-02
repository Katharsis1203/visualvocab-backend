# VisualVocab Training Manager — Phase 4

This update adds a backend training manager that merges all uploaded
YOLO-native datasets into one clean Ultralytics dataset.

## Replace in the backend repository

- `app.py`
- `templates/index.html`

## Add

- `templates/training.html`

Commit the files to the `main` branch. Render should redeploy automatically.

## Open

- Dashboard: `https://visualvocab-backend.onrender.com/`
- Training manager:
  `https://visualvocab-backend.onrender.com/visualvocab/training`

Click **Build combined dataset**.

The backend will:

1. Read every valid uploaded dataset ZIP.
2. Create a union of all class names.
3. Remap each source class ID into the combined class list.
4. Prefix filenames to avoid collisions.
5. Create a deterministic 80/20 train/validation split.
6. Produce:
   - `images/train/`
   - `images/val/`
   - `labels/train/`
   - `labels/val/`
   - `dataset.yaml`
   - `classes.json`
   - `build_metadata.json`

## Colab

Upload `VisualVocab_Train.ipynb` to Google Colab.

Before serious training, collect substantially more varied images. A tiny
dataset proves the pipeline but will not produce a robust model.

## Important

The combined ZIP is stored on Render's current filesystem. On a free Render
service, it can disappear after redeployment or instance replacement, just
like uploaded datasets.
