"""Model integration module.

Loads the TRAINED Keras model (model/tomato_model.keras) and runs real inference.
If the model file does not exist yet, ModelNotAvailable is raised -- no fake
predictions are ever produced. Train the model with:  python model/train.py
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

class ModelNotAvailable(Exception):
    pass

def slug_from_label(label: str) -> str:
    """Map a PlantVillage folder name (e.g. 'Tomato_Early_blight') to our info slug."""
    l = label.lower()
    rules = [("bacterial", "bacterial-spot"), ("early", "early-blight"), ("late", "late-blight"),
             ("mold", "leaf-mold"), ("septoria", "septoria-leaf-spot"), ("spider", "spider-mites"),
             ("target", "target-spot"), ("curl", "yellow-leaf-curl-virus"), ("yellow", "yellow-leaf-curl-virus"),
             ("mosaic", "mosaic-virus"), ("healthy", "healthy")]
    for key, slug in rules:
        if key in l:
            return slug
    raise ValueError(f"Unrecognised class label: {label}")

class Predictor:
    def __init__(self, model_path: Path, class_names_path: Path):
        self.model_path, self.class_names_path = Path(model_path), Path(class_names_path)
        self._model, self._slugs = None, None

    def available(self) -> bool:
        return self.model_path.exists() and self.class_names_path.exists()

    def _load(self):
        if self._model is not None:
            return
        if not self.available():
            raise ModelNotAvailable(
                "Trained model not found. Download the PlantVillage tomato data into dataset/tomato/ "
                "and run `python model/train.py` (see README.md).")
        import tensorflow as tf  # imported lazily so the site still starts without TF
        self._model = tf.keras.models.load_model(self.model_path)
        names = json.loads(self.class_names_path.read_text())
        self._slugs = [slug_from_label(n) for n in names]

    def predict(self, img: Image.Image, top_k: int = 3) -> list[dict]:
        """Resize to 224x224 and classify. Normalisation (scaling to [-1, 1], as MobileNetV2
        expects) is built into the saved model, so training and inference always match."""
        self._load()
        arr = np.asarray(img.convert("RGB").resize(IMG_SIZE), dtype="float32")[None, ...]
        probs = self._model.predict(arr, verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [{"slug": self._slugs[i], "confidence": round(float(probs[i]) * 100, 2)} for i in order]
