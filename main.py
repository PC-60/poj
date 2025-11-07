# main.py
import os
import io
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import joblib
import tensorflow as tf
from huggingface_hub import hf_hub_download


# -----------------------------
# Config / filenames
# -----------------------------
# Local filenames we will use after download
PCA_FILE = os.getenv("PCA_SVM_PATH", "pca_svm_model.joblib")
CNN_SVM_FILE = os.getenv("CNN_SVM_PATH", "cnn_svm_model.joblib")
CNN_KERAS_FILE = os.getenv("CNN_KERAS", "cnn_mri_model.keras")

# Hugging Face repo info (set HF_TOKEN only if the repo is private)
HF_REPO_ID = os.getenv("HF_REPO_ID", "your-username/brain-tumor-ai-models")
HF_REVISION = os.getenv("HF_REVISION", "main")
HF_TOKEN = os.getenv("HF_TOKEN", None)
IMG_SIZE = 224

# Globals filled at startup
pca_model = None
svc_model = None
label_map = None
resize_hw = None
label_list = None

feat_model = None
cnn_svm = None
label_map2 = None
label_list2 = None


# -----------------------------
# Small helpers
# -----------------------------
def hf_ensure(filename: str) -> str:
    """
    Ensure `filename` exists locally by downloading it from Hugging Face once.
    Returns the local path.
    """
    if Path(filename).exists():
        return filename
    path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="model",
        revision=HF_REVISION,
        token=HF_TOKEN,
        local_dir=".",
        local_dir_use_symlinks=False,
    )
    # hf_hub_download can return a cached path under ~/.cache; make a real copy
    if Path(path).resolve() != Path(filename).resolve():
        Path(filename).write_bytes(Path(path).read_bytes())
    return filename


def preprocess_for_flat(pil: Image.Image, hw):
    img = pil.convert("L").resize(hw)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()[None, :]


def preprocess_for_cnn(pil: Image.Image):
    img = pil.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=-1)  # 3-channels for TF models
    return arr[None, ...]


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://frontend-d0hdidl1r-parmeshwars-projects-16af44b0.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# -----------------------------
# Startup: download + load models
# -----------------------------
@app.on_event("startup")
def _startup():
    global pca_model, svc_model, label_map, resize_hw, label_list
    global feat_model, cnn_svm, label_map2, label_list2

    # 1) Make sure files exist locally (download from HF if missing)
    hf_ensure(PCA_FILE)
    hf_ensure(CNN_SVM_FILE)
    hf_ensure(CNN_KERAS_FILE)

    # 2) Load PCA+SVM bundle
    pca_bundle = joblib.load(PCA_FILE)
    pca_model = pca_bundle["pca"]
    svc_model = pca_bundle["svc"]
    label_map = pca_bundle["label_map"]
    resize_hw = pca_bundle.get("resize", (200, 200))
    label_list = [label_map[i] for i in sorted(label_map)]

    # 3) Load CNN (.keras) as feature extractor (layer named "feat")
    cnn = tf.keras.models.load_model(CNN_KERAS_FILE, compile=False)
    try:
        feat_layer = cnn.get_layer("feat")  # make sure your Dense layer is named "feat"
    except ValueError:
        # if layer name differs, print names to help debug
        raise ValueError(
            f'Layer "feat" not found. Available layers: {[l.name for l in cnn.layers]}'
        )
    feat_model = tf.keras.Model(cnn.input, feat_layer.output)

    # 4) Load SVM trained on CNN features
    cnn_svm_bundle = joblib.load(CNN_SVM_FILE)
    cnn_svm = cnn_svm_bundle["svm"]
    label_map2 = cnn_svm_bundle.get("label_map", label_map)
    label_list2 = [label_map2[i] for i in sorted(label_map2)]


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"ok": True, "models": ["pca_svm", "cnn_svm"]}


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), model: str = Form("pca_svm")):
    # Read image
    raw = await image.read()
    pil = Image.open(io.BytesIO(raw)).convert("RGB")

    if model == "cnn_svm":
        # CNN embeddings + SVM
        x = preprocess_for_cnn(pil)
        emb = feat_model.predict(x, verbose=0)
        probs = cnn_svm.predict_proba(emb)[0]
        idx = int(np.argmax(probs))
        return JSONResponse(
            {
                "prediction": label_list2[idx],
                "probability": float(probs[idx]),
                "probabilities": {
                    label_list2[i]: float(probs[i]) for i in range(len(probs))
                },
                "model": {"name": "CNN features + SVM (RBF)", "version": "1.0"},
            }
        )

    # Default: PCA + SVM
    x = preprocess_for_flat(pil, resize_hw)
    x_pca = pca_model.transform(x)
    if hasattr(svc_model, "predict_proba"):
        probs = svc_model.predict_proba(x_pca)[0]
        idx = int(np.argmax(probs))
        payload = {
            "prediction": label_list[idx],
            "probability": float(probs[idx]),
            "probabilities": {
                label_list[i]: float(probs[i]) for i in range(len(probs))
            },
            "model": {"name": "PCA + SVM", "version": "1.0"},
        }
    else:
        idx = int(svc_model.predict(x_pca)[0])
        payload = {
            "prediction": label_list[idx],
            "probability": None,
            "model": {"name": "PCA + SVM", "version": "1.0"},
        }
    return JSONResponse(payload)
