"""Detección automática de estructuras candidatas (huevos, quistes, trofozoítos).

No es un detector entrenado: usa visión por computador clásica para proponer
regiones con forma de huevo/quiste (ovaladas, de borde liso y tamaño plausible),
que después clasifica el modelo de la especie elegida.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

WORK_SIDE = 1024  # la detección se hace sobre una copia de este tamaño


@dataclass
class Candidate:
    box: tuple          # (x0, y0, x1, y1) en coordenadas de la imagen original
    score: float        # qué tan "ovalado y limpio" es el objeto (0-1)
    diameter: float     # diámetro equivalente como fracción del campo


def _field_mask(gray):
    """Zona útil de la imagen: todo lo que no es el negro del recorte del ocular."""
    m = np.where(gray > 8, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.full_like(gray, 255)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == big, 255, 0).astype(np.uint8)
    # Se erosiona para ignorar el borde del campo, que genera falsos objetos
    return cv2.erode(mask, np.ones((25, 25), np.uint8))


def _contained(a, b):
    """Fracción del recuadro más pequeño que queda dentro del otro."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / small if small else 0.0


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def detect_candidates(
    image: Image.Image,
    min_size: float = 0.02,
    max_size: float = 0.25,
    sensitivity: float = 0.5,
    max_regions: int = 20,
    padding: float = 0.35,
):
    """Devuelve candidatos ordenados por puntaje.

    min_size / max_size: diámetro mínimo y máximo del objeto, como fracción del
    lado mayor de la imagen. sensitivity (0-1): más alto = acepta formas menos
    regulares y contrastes más bajos (más candidatos, más falsos positivos).
    """
    rgb = np.array(image.convert("RGB"))
    H, W = rgb.shape[:2]
    scale = WORK_SIDE / max(H, W)
    small = cv2.resize(rgb, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    side = max(h, w)

    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    field = _field_mask(gray)

    # Diferencia de color respecto al fondo local (fondo = desenfoque grande)
    bg = cv2.GaussianBlur(lab, (0, 0), sigmaX=side * 0.06)
    diff = np.sqrt(((lab - bg) ** 2).sum(axis=2))
    diff[field == 0] = 0
    diff = cv2.GaussianBlur(diff, (0, 0), 1.5)

    vals = diff[field > 0]
    if vals.size == 0:
        return []

    # Varios umbrales de contraste: un huevo pegado a detritus se separa en alguno
    levels = sorted({max(float(np.percentile(vals, p)), 5.0) for p in (70, 76, 82, 88, 92, 95, 97, 98.5)})
    min_iou = 0.86 - 0.14 * sensitivity
    min_solidity = 0.93 - 0.13 * sensitivity
    kernel = np.ones((3, 3), np.uint8)

    # Mapas binarios: (a) contraste de color a varios umbrales y
    # (b) bordes cerrados y rellenos, para huevos de pared marcada e interior claro
    binaries = []
    for thr in levels:
        b = np.where(diff > thr, 255, 0).astype(np.uint8)
        binaries.append(cv2.morphologyEx(b, cv2.MORPH_OPEN, kernel))
    smooth = cv2.GaussianBlur(gray, (0, 0), 1.2)
    for lo, hi in ((20, 60), (35, 90)):
        edges = cv2.Canny(smooth, lo, hi)
        edges[field == 0] = 0
        edges = cv2.dilate(edges, kernel)
        cnts, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        filled_edges = np.zeros_like(edges)
        cv2.drawContours(filled_edges, cnts, -1, 255, cv2.FILLED)
        binaries.append(cv2.morphologyEx(filled_edges, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)))

    found = []
    for binary in binaries:
        contours, _ = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 25 or len(c) < 8:
                continue
            diameter = 2 * np.sqrt(area / np.pi) / side
            if not (min_size <= diameter <= max_size):
                continue
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity = area / hull_area if hull_area else 0
            if solidity < min_solidity:
                continue
            ellipse = cv2.fitEllipse(c)
            (cx, cy), (ax1, ax2), _ = ellipse
            if min(ax1, ax2) <= 0 or max(ax1, ax2) / min(ax1, ax2) > 3.2:
                continue

            # Qué tanto coincide el objeto con su elipse ajustada (IoU)
            x, y, bw, bh = cv2.boundingRect(c)
            m = int(max(bw, bh) * 0.3) + 2
            ox, oy = max(0, x - m), max(0, y - m)
            sw, sh = min(w, x + bw + m) - ox, min(h, y + bh + m) - oy
            obj = np.zeros((sh, sw), np.uint8)
            cv2.drawContours(obj, [c - [ox, oy]], -1, 1, cv2.FILLED)
            ell = np.zeros_like(obj)
            cv2.ellipse(ell, ((cx - ox, cy - oy), (ax1, ax2), ellipse[2]), 1, cv2.FILLED)
            inter = np.logical_and(obj, ell).sum()
            union = np.logical_or(obj, ell).sum()
            e_iou = inter / union if union else 0
            if e_iou < min_iou:
                continue

            contrast = float(diff[oy:oy + sh, ox:ox + sw][obj > 0].mean()) / (levels[-1] * 1.5)
            score = float(np.clip(0.6 * e_iou + 0.25 * solidity + 0.15 * min(contrast, 1.0), 0, 1))

            pad = padding * max(bw, bh)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            box = tuple(int(round(v / scale)) for v in (x0, y0, x1, y1))
            found.append(Candidate(box=box, score=score, diameter=float(diameter)))

    # (c) Círculos de pared continua (Hough), útil para huevos pegados a detritus
    r_min, r_max = max(5, int(min_size * side / 2)), max(8, int(max_size * side / 2))
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.HOUGH_GRADIENT_ALT, dp=1.5,
        minDist=max(10, r_min * 2), param1=120, param2=0.85 - 0.1 * sensitivity,
        minRadius=r_min, maxRadius=r_max,
    )
    if circles is not None:
        for cx, cy, r in circles[0]:
            if field[min(h - 1, int(cy)), min(w - 1, int(cx))] == 0:
                continue
            pad = padding * 2 * r
            x0, y0 = max(0, cx - r - pad), max(0, cy - r - pad)
            x1, y1 = min(w, cx + r + pad), min(h, cy + r + pad)
            box = tuple(int(round(v / scale)) for v in (x0, y0, x1, y1))
            found.append(Candidate(box=box, score=0.9, diameter=float(2 * r / side)))

    # Supresión de solapamientos: se queda con el de mayor puntaje
    found.sort(key=lambda c: c.score, reverse=True)
    kept = []
    for cand in found:
        if all(_iou(cand.box, k.box) < 0.3 and _contained(cand.box, k.box) < 0.7 for k in kept):
            kept.append(cand)
        if len(kept) >= max_regions:
            break
    return kept
