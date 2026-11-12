# Architecture
Browser (HTML/CSS/JS) --multipart--> Flask `/api/predict` --> validate (extension, size, Pillow verify)
--> `Predictor` (resize 224x224, Keras model with built-in Rescaling [-1,1]) --> softmax top-3
--> merge with `disease_info.json` --> JSON response; result saved to SQLite history + `uploads/`.
Training (offline): PlantVillage tomato -> 70/15/15 split -> augmentation -> MobileNetV2 (frozen) -> fine-tune top 30 layers -> test evaluation -> model + metrics saved.
