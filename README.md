# TomatoLeaf AI — Deep Learning-Based Tomato Leaf Disease Detection

Flask + TensorFlow/Keras (MobileNetV2 transfer learning) web app. Upload a tomato leaf image → predicted class
(10 categories), confidence, and disease information. **No fake predictions:** until you train the model, the
Detect page shows "Model not trained yet" and the API returns HTTP 503. No accuracy figure is shown until
`model/train.py` produces a real test-set evaluation.

## Structure
```
backend/        Flask app (app.py = API + serves site), predictor.py (model loading/inference), history.db (auto-created)
model/          train.py -> tomato_model.keras, class_names.json, metrics.json (created by training)
dataset/        put PlantVillage tomato folders in dataset/tomato/
disease_info/   disease_info.json (descriptions, symptoms, causes, prevention)
frontend/       HTML pages, css/, js/, img/diseases/ (optional photos named <slug>.jpg)
uploads/        analysed images
docs/           put your project report / screenshots here
```

## Setup (Windows / macOS / Linux, Python 3.9–3.12)
```bash
cd tomato-leaf-ai
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## 1. Place the dataset
Put the Kaggle PlantVillage tomato dataset here (folders with `train` and `val`, 10 class folders in each):
```
tomato-leaf-ai/dataset/tomato/train/Tomato___healthy/...        (and the other 9 class folders)
tomato-leaf-ai/dataset/tomato/val/Tomato___healthy/...          (the same 10 class folders)
```
Check it first (no training happens):
```bash
python model/train.py --dry-run
```
This verifies all 10 classes in both folders, removes unreadable files and exact duplicate images, and prints class counts and the 70/15/15 stratified split.

## 2. Train the model
```bash
python model/train.py --epochs 10 --finetune-epochs 5
```
Train/val images are merged, de-duplicated, then re-split 70/15/15 (stratified, seed 42). It saves
`model/tomato_model.keras`, `model/class_names.json`, `model/metrics.json` (accuracy, precision, recall, F1, confusion matrix,
classification report, dataset counts) and `model/splits.json` (which image went to which split).

## 3. Run the website
```bash
python backend/app.py
```
Open http://127.0.0.1:5000

## API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/predict` | multipart field `image` (JPG/PNG/WEBP, ≤5 MB) → prediction, confidence, top-3, disease info |
| GET | `/api/status` | whether a trained model is loaded |
| GET | `/api/diseases` | all disease info |
| GET | `/api/history`, DELETE `/api/history` | prediction history |
| GET | `/api/stats` | per-class counts |
| GET | `/api/model-metrics` | real metrics from training |
## Model Performance and Evaluation

The model was evaluated using a separate test dataset. The evaluation metrics, including accuracy, precision, recall, and F1-score, are available in `model/metrics.json`.

## Dataset and Disease Information

The project uses the PlantVillage tomato leaf dataset for training and evaluation. Disease descriptions and reference images are available in the `disease_info/` and `frontend/img/diseases/` directories.

## Troubleshooting

- **Model not found:** Verify that the trained model file exists in the `model/` directory.
- **TensorFlow installation issues:** Use a compatible 64-bit Python version (3.10–3.12).
- **Low prediction confidence:** Use clear images of individual tomato leaves with good lighting for better predictions.


