# Plant Doctor 🌿

ML-powered plant health diagnosis from leaf photos. Trained on the PlantVillage dataset (54k images, 38 classes of healthy + diseased leaves across 14 plant species).

## Status

- ✅ Phase 1: Data setup + EDA
- ✅ Phase 2: Model training (MobileNetV3, **99.3% test accuracy**)
- ✅ Phase 3: FastAPI backend
- ✅ Phase 4: React PWA frontend
- 🚧 Phase 5: Deployment

## Tech stack

- **ML**: PyTorch, torchvision (MobileNetV3-Small with transfer learning)
- **Data**: PlantVillage dataset via Kaggle API
- **Tooling**: Jupyter, scikit-learn, pandas, React, Vite, Tailwind CSS v4

## Project structure

\`\`\`
src/         # Reusable modules (dataset, model, training, evaluation)
notebooks/   # Exploration and experiments
data/        # Datasets (gitignored)
models/      # Trained model weights (gitignored)
api/         # FastAPI backend (coming soon)
web/         # React + Vite + Tailwind frontend with PWA support
\`\`\`

## Setup

\`\`\`bash
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
\`\`\`

## Results

| Model | Test accuracy | Trainable params |
|-------|---------------|-------------------|
| SimpleCNN (baseline) | ~83% | 98k |
| MobileNetV3-Small (transfer learning) | **99.3%** | 1.5M |
