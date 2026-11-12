"""Flask backend: serves the website and the prediction API."""
import io, json, sqlite3, uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from .predictor import Predictor, ModelNotAvailable

ROOT = Path(__file__).resolve().parent.parent
UPLOADS, FRONTEND = ROOT / "uploads", ROOT / "frontend"
DB = Path(__file__).resolve().parent / "history.db"
ALLOWED = {"jpg", "jpeg", "png", "webp"}
MAX_MB = 5
LOW_CONFIDENCE = 60.0  # below this, the UI warns the user

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
INFO = {d["slug"]: d for d in json.loads((ROOT / "disease_info/disease_info.json").read_text(encoding="utf-8"))}
predictor = Predictor(ROOT / "model/tomato_model.keras", ROOT / "model/class_names.json")

def db():
    con = sqlite3.connect(DB); con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY, filename TEXT, slug TEXT,
                   confidence REAL, created TEXT)""")
    return con

def err(msg, code): return jsonify({"error": msg}), code

@app.errorhandler(413)
def too_big(_): return err(f"File too large. Maximum size is {MAX_MB} MB.", 413)

@app.get("/api/status")
def status():
    return jsonify({"model_available": predictor.available(), "max_mb": MAX_MB, "allowed": sorted(ALLOWED)})

@app.get("/api/diseases")
def diseases(): return jsonify(list(INFO.values()))

@app.get("/api/model-metrics")
def metrics():
    p = ROOT / "model/metrics.json"   # written by train.py from a real test-set evaluation
    return jsonify(json.loads(p.read_text())) if p.exists() else (jsonify({"available": False}), 404)

@app.post("/api/predict")
def predict():
    f = request.files.get("image")
    if not f or not f.filename: return err("No image uploaded.", 400)
    ext = f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED: return err("Unsupported format. Use JPG, PNG or WEBP.", 415)
    data = f.read()
    try:
        img = Image.open(io.BytesIO(data)); img.verify(); img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        return err("The file is not a valid image.", 400)
    try:
        top = predictor.predict(img)
    except ModelNotAvailable as e:
        return err(str(e), 503)
    best = top[0]; info = INFO[best["slug"]]
    rid = uuid.uuid4().hex; fname = f"{rid}.{ext}"
    (UPLOADS / fname).write_bytes(data)
    created = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        con.execute("INSERT INTO history VALUES(?,?,?,?,?)", (rid, fname, best["slug"], best["confidence"], created))
    return jsonify({
        "id": rid, "timestamp": created, "image_url": f"/uploads/{fname}",
        "prediction": {"slug": best["slug"], "name": info["name"], "confidence": best["confidence"],
                       "is_healthy": info["healthy"], "low_confidence": best["confidence"] < LOW_CONFIDENCE},
        "top_predictions": [{"name": INFO[t["slug"]]["name"], "confidence": t["confidence"]} for t in top],
        "info": info})

@app.get("/api/history")
def history():
    with db() as con: rows = con.execute("SELECT * FROM history ORDER BY created DESC LIMIT 100").fetchall()
    return jsonify([{**dict(r), "name": INFO[r["slug"]]["name"], "image_url": f"/uploads/{r['filename']}",
                     "is_healthy": INFO[r["slug"]]["healthy"]} for r in rows])

@app.delete("/api/history")
def clear_history():
    with db() as con: con.execute("DELETE FROM history")
    for p in UPLOADS.glob("*"):
        if p.name != ".gitkeep": p.unlink()
    return jsonify({"ok": True})

@app.get("/api/stats")
def stats():
    with db() as con:
        rows = con.execute("SELECT slug, COUNT(*) n, AVG(confidence) avg FROM history GROUP BY slug").fetchall()
    per = {r["slug"]: {"count": r["n"], "avg_confidence": round(r["avg"], 2)} for r in rows}
    total = sum(v["count"] for v in per.values())
    return jsonify({"total": total, "per_class": [
        {"slug": s, "name": i["name"], **per.get(s, {"count": 0, "avg_confidence": 0})} for s, i in INFO.items()]})

@app.get("/uploads/<path:name>")
def uploads(name): return send_from_directory(UPLOADS, name)

@app.get("/")
def home(): return send_from_directory(FRONTEND, "index.html")

@app.get("/<path:path>")
def static_files(path): return send_from_directory(FRONTEND, path)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
