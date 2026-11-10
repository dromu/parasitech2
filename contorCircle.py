import cv2
import numpy as np
from PIL import Image


def countourCircle(imagenPil, umbralMen):
    """Recorta el campo circular del microscopio usando el contorno más grande.

    Si no se encuentra ningún contorno con el umbral dado, devuelve la imagen
    original (en RGB) en lugar de fallar.
    """
    image = np.array(imagenPil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Desenfoque para reducir ruido y umbral para obtener una imagen binaria
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, umbralMen, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Máscara con el contorno más grande y recorte de su rectángulo delimitador
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    x, y, w, h = cv2.boundingRect(largest_contour)
    result = masked_image[y:y + h, x:x + w]
    if result.size == 0 or w < 32 or h < 32:
        result = image

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
