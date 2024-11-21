import cv2
import numpy as np
from PIL import Image

def countourCircle(imagenPil, umbralMen):
    # Convertir la imagen PIL a NumPy array
    image = np.array(imagenPil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar un desenfoque para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un umbral para crear una imagen binaria
    _, thresholded = cv2.threshold(blurred, umbralMen, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Seleccionar el contorno más grande
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Crear una máscara con el contorno más grande
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
        
        # Aplicar la máscara a la imagen original
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Encontrar el rectángulo delimitador del contorno más grande
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Recortar la región de interés
        result = masked_image[y:y+h, x:x+w]

    # Convertir de BGR a RGB para PIL
    imagen_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    imagen_pil = Image.fromarray(imagen_rgb)
    
    return imagen_pil
