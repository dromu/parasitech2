"""Sube los pesos locales a Hugging Face Hub (ejecutar UNA vez, en tu PC).

Uso:
    pip install huggingface_hub
    huggingface-cli login            # pega un token con permiso de escritura
    git lfs pull                     # si aún tienes los .pkl como punteros LFS
    python scripts/upload_models.py  # o: PARASITECH_HF_REPO=usuario/repo python ...
"""
import os
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = os.environ.get("PARASITECH_HF_REPO", "dromu/parasitech-models")
FILES = ["efficiennetB5_90perce.pkl", "vit_small_patch16_224_Helmintos.pkl"]

api = HfApi()
api.create_repo(REPO_ID, repo_type="model", exist_ok=True)  # private=True si lo quieres privado

for name in FILES:
    path = Path("models") / name
    if path.stat().st_size < 1_000_000:
        raise SystemExit(f"{path} parece un puntero de Git LFS; ejecuta 'git lfs pull' primero.")
    print(f"Subiendo {path} -> {REPO_ID}")
    api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=REPO_ID)

print("Listo.")
