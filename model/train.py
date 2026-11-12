"""Train the tomato leaf classifier (MobileNetV2 transfer learning) on the Kaggle PlantVillage tomato dataset.

Expected dataset layout (as downloaded from Kaggle):
    dataset/tomato/train/<10 class folders>/*.jpg
    dataset/tomato/val/<the same 10 class folders>/*.jpg

What this script does:
  1. reads images from BOTH train/ and val/, verifies the 10 classes exist in both
  2. drops unreadable files and exact duplicate images (SHA-256 of file content)
  3. makes a stratified 70/15/15 train/validation/test split with a fixed seed
  4. trains (frozen backbone) and fine-tunes MobileNetV2; augmentation is active only while training
  5. evaluates once on the untouched test split and saves REAL metrics

Usage (from the project folder):
    python model/train.py --dry-run                       # dataset checks + counts only, no training
    python model/train.py --epochs 10 --finetune-epochs 5
Outputs in model/: tomato_model.keras, class_names.json, metrics.json, splits.json
"""
import argparse, hashlib, io, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "backend"))
from predictor import slug_from_label            # the exact mapping the Flask backend uses

SIZE = (224, 224)
EXTS = {".jpg", ".jpeg", ".png"}
N_CLASSES = 10
# Dataset folder name -> disease_info.json slug. Anything not listed falls back to the backend's keyword mapping.
FOLDER_TO_SLUG = {
    "Tomato___Bacterial_spot": "bacterial-spot",
    "Tomato___Early_blight": "early-blight",
    "Tomato___Late_blight": "late-blight",
    "Tomato___Leaf_Mold": "leaf-mold",
    "Tomato___Septoria_leaf_spot": "septoria-leaf-spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "spider-mites",
    "Tomato___Target_Spot": "target-spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "yellow-leaf-curl-virus",
    "Tomato___Tomato_mosaic_virus": "mosaic-virus",
    "Tomato___healthy": "healthy",
}

# ---------------------------------------------------------------- dataset handling
def find_split_dirs(data_dir: Path):
    train = data_dir / "train"
    val = next((data_dir / n for n in ("val", "valid", "validation") if (data_dir / n).is_dir()), None)
    if not train.is_dir() or val is None:
        sys.exit(f"ERROR: expected '{data_dir}' to contain 'train' and 'val' folders.\n"
                 f"Found: {[p.name for p in data_dir.iterdir()] if data_dir.is_dir() else 'folder does not exist'}")
    return train, val

def class_folders(d: Path):
    return sorted(p.name for p in d.iterdir() if p.is_dir())

def verify_classes(train_dir: Path, val_dir: Path):
    tr, va = set(class_folders(train_dir)), set(class_folders(val_dir))
    problems = []
    if tr != va:
        problems.append(f"classes differ. Only in train: {sorted(tr - va)}; only in val: {sorted(va - tr)}")
    if len(tr) != N_CLASSES:
        problems.append(f"expected {N_CLASSES} classes in train, found {len(tr)}: {sorted(tr)}")
    if problems:
        sys.exit("ERROR: " + " | ".join(problems))
    return sorted(tr)

def map_labels(names):
    """Check every folder maps to a unique, known disease_info slug (keeps backend/frontend compatible)."""
    known = {d["slug"] for d in json.loads((ROOT / "disease_info/disease_info.json").read_text(encoding="utf-8"))}
    slugs = []
    for n in names:
        s = FOLDER_TO_SLUG.get(n) or slug_from_label(n)
        if slug_from_label(n) != s:
            sys.exit(f"ERROR: mapping for '{n}' disagrees with backend/predictor.py ({s} vs {slug_from_label(n)})")
        slugs.append(s)
    if len(set(slugs)) != len(slugs) or set(slugs) != known:
        sys.exit(f"ERROR: class mapping does not match disease_info.json. Mapped: {slugs}")
    return slugs

def scan_images(split_dir: Path, names):
    """Return records (sha256, path, class_index) for every readable image; also unreadable files."""
    recs, bad = [], []
    for ci, cls in enumerate(names):
        for p in sorted((split_dir / cls).rglob("*")):
            if p.suffix.lower() not in EXTS or not p.is_file():
                continue
            try:
                data = p.read_bytes()
                Image.open(io.BytesIO(data)).verify()
                recs.append((hashlib.sha256(data).hexdigest(), p, ci))
            except Exception:
                bad.append(p)
    return recs, bad

def build_dataset(data_dir: Path, names):
    train_dir, val_dir = find_split_dirs(data_dir)
    print("Scanning images (hashing to detect duplicates)...")
    r_tr, bad_tr = scan_images(train_dir, names)
    r_va, bad_va = scan_images(val_dir, names)
    raw = {"train": np.bincount([r[2] for r in r_tr], minlength=len(names)).tolist(),
           "val": np.bincount([r[2] for r in r_va], minlength=len(names)).tolist()}
    groups = defaultdict(list)
    for rec in r_tr + r_va:                      # train first so its copy is the one kept
        groups[rec[0]].append(rec)
    kept, dup_removed, conflicts = [], 0, 0
    for h, g in groups.items():
        if len({r[2] for r in g}) > 1:           # same image under different labels -> unreliable, drop all
            conflicts += len(g); continue
        kept.append(g[0]); dup_removed += len(g) - 1
    kept.sort(key=lambda r: str(r[1]))
    paths = [r[1] for r in kept]; y = np.array([r[2] for r in kept])
    info = {"raw_counts": raw, "unreadable_files": len(bad_tr) + len(bad_va),
            "duplicates_removed": dup_removed, "conflicting_label_images_removed": conflicts,
            "unique_images": len(paths)}
    return paths, y, info

def stratified_split(n, y, seed):
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    return tr, va, te

def report_dataset(names, y, info, splits):
    cnt = lambda ids: np.bincount(y[ids], minlength=len(names))
    tr, va, te = (cnt(s) for s in splits)
    tot = cnt(np.arange(len(y)))
    w = max(len(n) for n in names)
    print(f"\nRaw images found: train/={sum(info['raw_counts']['train'])}, val/={sum(info['raw_counts']['val'])}")
    print(f"Unreadable files skipped: {info['unreadable_files']} | exact duplicates removed: {info['duplicates_removed']} "
          f"| same image under different labels removed: {info['conflicting_label_images_removed']}")
    print(f"Unique images used: {info['unique_images']}\n")
    print(f"{'Class'.ljust(w)} {'Raw train':>9} {'Raw val':>8} {'Unique':>7} {'Share':>7} {'Train':>6} {'Val':>5} {'Test':>5}")
    for i, n in enumerate(names):
        print(f"{n.ljust(w)} {info['raw_counts']['train'][i]:>9} {info['raw_counts']['val'][i]:>8} {tot[i]:>7} "
              f"{tot[i] / tot.sum() * 100:>6.1f}% {tr[i]:>6} {va[i]:>5} {te[i]:>5}")
    print(f"{'TOTAL'.ljust(w)} {sum(info['raw_counts']['train']):>9} {sum(info['raw_counts']['val']):>8} {tot.sum():>7} "
          f"{100:>6.1f}% {tr.sum():>6} {va.sum():>5} {te.sum():>5}")
    print(f"Imbalance ratio (largest/smallest class): {tot.max() / tot.min():.1f}\n")
    return {**info, "unique_per_class": dict(zip(names, map(int, tot))),
            "split_sizes": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "split_per_class": {n: {"train": int(tr[i]), "val": int(va[i]), "test": int(te[i])} for i, n in enumerate(names)},
            "imbalance_ratio": round(float(tot.max() / tot.min()), 2)}

# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "dataset" / "tomato"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--finetune-epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="only check the dataset, print counts and splits; no training")
    a = ap.parse_args()
    data_dir = Path(a.data_dir)

    train_dir, val_dir = find_split_dirs(data_dir)
    names = verify_classes(train_dir, val_dir)
    slugs = map_labels(names)
    print(f"All {N_CLASSES} classes found in both train/ and val/, and mapped to the website's disease information:")
    for n, s in zip(names, slugs): print(f"   {n}  ->  {s}")
    paths, y, info = build_dataset(data_dir, names)
    splits = stratified_split(len(paths), y, a.seed)
    dataset_report = report_dataset(names, y, info, splits)
    if a.dry_run:
        print("Dry run finished: nothing was trained or saved."); return

    import tensorflow as tf                       # imported late so --dry-run is fast
    from tensorflow import keras
    from tensorflow.keras import layers
    from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                 precision_recall_fscore_support)
    keras.utils.set_random_seed(a.seed)
    tr_i, va_i, te_i = splits
    P = np.array([str(p) for p in paths])

    def load(path):                                # identical to backend/predictor.py: RGB + resize to 224x224
        p = path.decode("utf-8") if isinstance(path, bytes) else str(path)
        return np.asarray(Image.open(p).convert("RGB").resize(SIZE), dtype="float32")

    def make_ds(ids, training):
        ds = tf.data.Dataset.from_tensor_slices((P[ids], y[ids].astype("int32")))
        if training: ds = ds.shuffle(len(ids), seed=a.seed, reshuffle_each_iteration=True)
        def f(p, l):
            x = tf.numpy_function(load, [p], tf.float32); x.set_shape(SIZE + (3,)); return x, l
        return ds.map(f, num_parallel_calls=tf.data.AUTOTUNE).batch(a.batch).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds, test_ds = make_ds(tr_i, True), make_ds(va_i, False), make_ds(te_i, False)

    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=SIZE + (3,), pooling="avg")
    base.trainable = False
    inp = keras.Input(SIZE + (3,))
    x = keras.Sequential([layers.RandomFlip("horizontal_and_vertical"), layers.RandomRotation(0.15),
                          layers.RandomZoom(0.15), layers.RandomContrast(0.1)], name="augmentation")(inp)  # active only in training
    x = layers.Rescaling(1 / 127.5, offset=-1, name="normalize")(x)      # same as MobileNetV2 preprocess_input
    x = base(x, training=False)
    x = layers.Dropout(0.3)(x)
    model = keras.Model(inp, layers.Dense(len(names), activation="softmax")(x))

    cbs = lambda: [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
    t0 = time.time()
    print("\n--- Phase 1: training the classifier head (backbone frozen) ---")
    model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs())
    print("\n--- Phase 2: fine-tuning the top 30 backbone layers ---")
    base.trainable = True
    for l in base.layers[:-30]: l.trainable = False
    model.compile(keras.optimizers.Adam(1e-5), "sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=a.finetune_epochs, callbacks=cbs())

    print("\n--- Evaluating on the independent test split ---")
    y_true = y[te_i]
    y_pred = model.predict(test_ds, verbose=0).argmax(1)
    loss, _ = model.evaluate(test_ds, verbose=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    labels = list(range(len(names)))
    metrics = {"available": True, "architecture": "MobileNetV2 (transfer learning, ImageNet weights)",
               "test_samples": int(len(y_true)), "test_accuracy": round(float(acc), 4), "test_loss": round(float(loss), 4),
               "macro": {"precision": round(float(pm), 4), "recall": round(float(rm), 4), "f1": round(float(fm), 4)},
               "weighted": {"precision": round(float(pw), 4), "recall": round(float(rw), 4), "f1": round(float(fw), 4)},
               "classes": names, "class_slugs": dict(zip(names, slugs)),
               "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
               "report": classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0),
               "dataset": dataset_report,
               "training": {"seed": a.seed, "batch_size": a.batch, "epochs_requested": [a.epochs, a.finetune_epochs],
                            "epochs_run": [len(h1.history["loss"]), len(h2.history["loss"])],
                            "history_phase1": {k: [round(float(v), 5) for v in vs] for k, vs in h1.history.items()},
                            "history_phase2": {k: [round(float(v), 5) for v in vs] for k, vs in h2.history.items()},
                            "minutes": round((time.time() - t0) / 60, 1)}}
    print(classification_report(y_true, y_pred, labels=labels, target_names=names, digits=4, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted):")
    print(np.array(metrics["confusion_matrix"]))

    rel = lambda ids: [Path(p).relative_to(data_dir).as_posix() for p in P[ids]]
    model.save(OUT / "tomato_model.keras")
    (OUT / "class_names.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "splits.json").write_text(json.dumps({"seed": a.seed, "train": rel(tr_i), "val": rel(va_i), "test": rel(te_i)}), encoding="utf-8")
    print(f"\nSaved tomato_model.keras, class_names.json, metrics.json, splits.json in {OUT}")
    print(f"Test accuracy: {acc * 100:.2f}%  |  macro F1: {fm:.4f}")

if __name__ == "__main__":
    main()
