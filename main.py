# main.py
import os, io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download

# ---------- keep TF light on tiny instances ----------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# If you add these two as Render env vars too, TF respects them; also set here as fallback:
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ---------- filenames & HF repo ----------
PCA_FILE = os.getenv("PCA_SVM_PATH", "pca_svm_model.joblib")
CNN_SVM_FILE = os.getenv("CNN_SVM_PATH", "cnn_svm_model.joblib")
CNN_KERAS_FILE = os.getenv("CNN_KERAS", "cnn_mri_model.keras")

HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/brain-tumor-ai-models")
HF_REVISION = os.getenv("HF_REVISION", "main")
HF_TOKEN = os.getenv("HF_TOKEN") or None   # important: blank -> None

IMG_SIZE = 224

# ---------- lazy globals (None until first use) ----------
_pca_bundle = None      # dict with pca, svc, label_map, resize
_cnn_models = None      # dict with feat_model, svm, label_map2

def hf_ensure(filename: str) -> str:
    """Download file from HF to local file with same name if missing. Return local path."""
    if Path(filename).exists():
        return filename
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="model",
        revision=HF_REVISION,
        token=HF_TOKEN,
        local_dir=".",
        local_dir_use_symlinks=False,   # ignored but safe
    )
    # copy from cache to project root if needed
    p = Path(filename)
    if Path(path).resolve() != p.resolve():
        p.write_bytes(Path(path).read_bytes())
    return filename

def load_pca_bundle():
    """Load PCA+SVM on demand (small)."""
    global _pca_bundle
    if _pca_bundle is not None:
        return _pca_bundle

    hf_ensure(PCA_FILE)
    bundle = joblib.load(PCA_FILE)
    # normalize expected keys
    pca = bundle["pca"]
    svc = bundle["svc"]
    label_map = bundle["label_map"]
    resize = bundle.get("resize", (200, 200))
    label_list = [label_map[i] for i in sorted(label_map)]
    _pca_bundle = dict(pca=pca, svc=svc, label_map=label_map, resize=resize, label_list=label_list)
    return _pca_bundle

def load_cnn_bundle():
    """Load Keras feature extractor + SVM on demand (heavy)."""
    global _cnn_models
    if _cnn_models is not None:
        return _cnn_models

    # ensure files exist locally
    hf_ensure(CNN_KERAS_FILE)
    hf_ensure(CNN_SVM_FILE)

    # load keras model lazily
    cnn = tf.keras.models.load_model(CNN_KERAS_FILE, compile=False)

    # try to get the 'feat' layer; if not found, fall back to penultimate layer
    try:
        feat_layer = cnn.get_layer("feat")
    except ValueError:
        feat_layer = cnn.layers[-2]

    feat_model = tf.keras.Model(cnn.input, feat_layer.output)

    # load SVM
    bundle = joblib.load(CNN_SVM_FILE)
    svm = bundle["svm"]
    label_map2 = bundle.get("label_map", {0: "No Tumor", 1: "Pituitary Tumor", 2: "Glioma Tumor", 3: "Meningioma Tumor"})
    label_list2 = [label_map2[i] for i in sorted(label_map2)]
    _cnn_models = dict(feat_model=feat_model, svm=svm, label_map2=label_map2, label_list2=label_list2)
    return _cnn_models

def preprocess_for_flat(pil: Image.Image, hw):
    img = pil.convert("L").resize(hw)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()[None, :]

def preprocess_for_cnn(pil: Image.Image):
    img = pil.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=-1)
    return arr[None, ...]

# ---------- FastAPI ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://frontend-d0hdidl1r-parmeshwars-projects-16af44b0.vercel.app",
    ],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=False
)

@app.get("/")
def root():
    return {"ok": True, "models": ["pca_svm", "cnn_svm"]}

@app.post("/api/predict")
async def predict(image: UploadFile = File(...), model: str = Form("pca_svm")):
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    if model == "cnn_svm":
        cnnb = load_cnn_bundle()  # may load once, heavy
        x = preprocess_for_cnn(pil)
        emb = cnnb["feat_model"].predict(x, verbose=0)
        probs = cnnb["svm"].predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        labels = cnnb["label_list2"]
        return JSONResponse({
            "prediction": labels[idx],
            "probability": float(probs[idx]),
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(probs))},
            "model": {"name": "CNN features + SVM (RBF)", "version": "1.0"}
        })

    # default: PCA + SVM
    p = load_pca_bundle()  # tiny
    x = preprocess_for_flat(pil, p["resize"])
    x_pca = p["pca"].transform(x)
    svc = p["svc"]
    labels = p["label_list"]
    if hasattr(svc, "predict_proba"):
        probs = svc.predict_proba(x_pca)[0]
        idx = int(np.argmax(probs))
        payload = {
            "prediction": labels[idx],
            "probability": float(probs[idx]),
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(probs))},
            "model": {"name": "PCA + SVM", "version": "1.0"}
        }
    else:
        idx = int(svc.predict(x_pca)[0])
        payload = {
            "prediction": labels[idx],
            "probability": None,
            "model": {"name": "PCA + SVM", "version": "1.0"}
        }
    return JSONResponse(payload)
