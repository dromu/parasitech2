from fastai.vision.all import *
from PIL import Image
from pathlib import Path
from pathlib import PosixPath



# Cargar el modelo globalmente para evitar cargarlo repetidamente
pathModel = Path("efficiennetB5_90perce.pkl")
learn = load_learner(pathModel)


pathModelH = Path("vit_small_patch16_224_Helmintos.pkl")
learnHel = load_learner(pathModelH)

def predictParasite( img,parasite): 

    # Realizar la predicción según el tipo de parásito
    if parasite == "Protozoo": 
        pred_class, pred_idx, outputs = learn.predict(img)
        return pred_class, pred_idx, outputs  # Devolver el resultado de la predicción

    elif parasite == "Helminto":
        pred_class, pred_idx, outputs = learnHel.predict(img)
        return pred_class, pred_idx, outputs  # Devolver el resultado de la predicción

    else:
        raise ValueError("Tipo de parásito no reconocido. Usa 'protozoo' o 'helminto'.")


