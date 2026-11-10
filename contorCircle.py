import cv2
import numpy as np
from PIL import Image

ANALYSIS_SIDE = 768  # tamaño de trabajo para encontrar el campo


def segment_field(imagen_pil):
    """Recorta automáticamente el campo circular del ocular del microscopio.

    Si la foto se tomó a través del ocular (campo iluminado rodeado de zona
    oscura: tubo, cámara, mesa), se detecta el círculo, se pone en negro todo
    lo que queda fuera y se recorta al campo. Si la imagen ya es un campo
    completo (sin borde oscuro), se devuelve tal cual.

    Devuelve (imagen_PIL_RGB, se_recorto: bool).
    """
    rgb = np.array(imagen_pil.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = ANALYSIS_SIDE / max(H, W)
    small = cv2.resize(rgb, (max(1, int(W * scale)), max(1, int(H * scale))), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]

    # Brillo: el campo iluminado es la región clara más grande
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (0, 0), 3)
    otsu, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Si casi no hay contraste entre "claro" y "oscuro", no hay borde de ocular
    bright_mean = gray[mask > 0].mean() if (mask > 0).any() else 0
    dark_mean = gray[mask == 0].mean() if (mask == 0).any() else 255
    if bright_mean - dark_mean < 45:
        return Image.fromarray(rgb), False

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return Image.fromarray(rgb), False
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    (cx, cy), r = cv2.minEnclosingCircle(largest)

    # Debe parecerse a un círculo: ocupa buena parte de su círculo envolvente
    circle_area = np.pi * r * r
    fill = area / circle_area if circle_area else 0
    frac = area / (h * w)
    # La zona fuera del círculo debe ser mayoritariamente oscura
    outside = np.ones((h, w), np.uint8)
    cv2.circle(outside, (int(cx), int(cy)), int(r), 0, -1)
    out_dark = (gray[outside > 0] < otsu).mean() if (outside > 0).any() else 0

    if fill < 0.72 or not (0.08 <= frac <= 0.92) or out_dark < 0.6:
        return Image.fromarray(rgb), False

    # Círculo a resolución completa, ligeramente reducido para quitar el borde difuso
    CX, CY, R = cx / scale, cy / scale, r / scale * 0.985
    full_mask = np.zeros((H, W), np.uint8)
    cv2.circle(full_mask, (int(round(CX)), int(round(CY))), int(round(R)), 255, -1)
    out = cv2.bitwise_and(rgb, rgb, mask=full_mask)
    x0, y0 = max(0, int(CX - R)), max(0, int(CY - R))
    x1, y1 = min(W, int(CX + R) + 1), min(H, int(CY + R) + 1)
    return Image.fromarray(out[y0:y1, x0:x1]), True


def countourCircle(imagenPil, umbralMen=None):
    """Compatibilidad con el código anterior: ahora el recorte es automático."""
    return segment_field(imagenPil)[0]
