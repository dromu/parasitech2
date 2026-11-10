"""Carga perezosa de los modelos de ParasiTech.

Los pesos (.pkl) ya NO viven en el repositorio de GitHub. Se descargan bajo
demanda desde Hugging Face Hub (y quedan en la caché local de huggingface_hub),
y solo se mantiene UN modelo en memoria a la vez.

Configuración (variables de entorno o secrets de Streamlit Cloud):
    PARASITECH_HF_REPO  -> repo de modelos en HF (por defecto "dromu/parasitech-models")
    HF_TOKEN            -> solo si el repo de modelos es privado
"""
import os
from pathlib import Path

import streamlit as st
from fastai.vision.all import *  # noqa: F401,F403  (necesario para deserializar los learners)
from huggingface_hub import hf_hub_download

HF_REPO_ID = os.environ.get("PARASITECH_HF_REPO", "dromu/parasitech-models")
LOCAL_MODELS_DIR = Path("models")

MODEL_FILES = {
    "Protozoo": "efficiennetB5_90perce.pkl",
    "Helminto": "vit_small_patch16_224_Helmintos.pkl",
}


def _model_path(filename: str) -> Path:
    """Usa la copia local si existe (desarrollo); si no, descarga desde HF Hub."""
    local = LOCAL_MODELS_DIR / filename
    # > 1 MB para ignorar punteros de Git LFS que hayan quedado en disco
    if local.exists() and local.stat().st_size > 1_000_000:
        return local
    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            token=os.environ.get("HF_TOKEN"),
        )
    )


@st.cache_resource(max_entries=1, show_spinner="Cargando modelo...")
def get_learner(parasite: str):
    """Carga el modelo pedido. max_entries=1 libera el anterior al cambiar de especie."""
    if parasite not in MODEL_FILES:
        raise ValueError("Tipo de parásito no reconocido. Usa 'Protozoo' o 'Helminto'.")
    return load_learner(_model_path(MODEL_FILES[parasite]), cpu=True)


def predictParasite(img, parasite):
    pred_class, pred_idx, outputs = get_learner(parasite).predict(img)
    return pred_class, pred_idx, outputs


def predict_batch(imgs, parasite, k=3):
    """Clasifica varias imágenes de una vez. Devuelve una lista de top-k por imagen."""
    if not imgs:
        return []
    learn = get_learner(parasite)
    vocab = list(learn.dls.vocab)
    try:
        dl = learn.dls.test_dl(list(imgs), bs=len(imgs), num_workers=0)
        with learn.no_bar():
            probs, _ = learn.get_preds(dl=dl)
    except Exception as e:  # respaldo: una por una
        print(f"[ParasiTech] Lote falló ({e}); clasificando una por una")
        return [predict_topk(img, parasite, k) for img in imgs]
    out = []
    for row in probs:
        top = row.topk(min(k, len(vocab)))
        out.append([(str(vocab[int(i)]), float(p)) for p, i in zip(top.values, top.indices)])
    return out


def predict_topk(img, parasite, k=3):
    """Devuelve las k clases más probables como [(clase, probabilidad 0-1), ...]."""
    learn = get_learner(parasite)
    _, _, probs = learn.predict(img)
    vocab = list(learn.dls.vocab)
    top = probs.topk(min(k, len(vocab)))
    return [(str(vocab[int(i)]), float(p)) for p, i in zip(top.values, top.indices)]
