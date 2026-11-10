"""Nombres científicos e imagen de salida (reporte) de ParasiTech."""
import os
import re
import unicodedata
from datetime import date
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

BOX_COLOR = (224, 17, 95)         # #E0115F
BAND_BG = (0, 0, 0)
TEXT = (255, 255, 255)
MUTED = (170, 170, 170)
WATERMARK = "staticImages/parasitechWatermark.png"
DISCLAIMER = ("Resultado generado con inteligencia artificial como apoyo al diagnóstico. "
              "Puede contener errores y debe ser confirmado por personal experto.")

# --------------------------------------------------------------------------- #
# Nombres científicos
# --------------------------------------------------------------------------- #
# clave normalizada -> (parte en cursiva, sufijo en redonda)
_SCI = {
    "ascaris": ("Ascaris lumbricoides", ""),
    "ascarislumbricoides": ("Ascaris lumbricoides", ""),
    "asclum": ("Ascaris lumbricoides", ""),
    "trichuris": ("Trichuris trichiura", ""),
    "trichuristrichiura": ("Trichuris trichiura", ""),
    "tricocefalo": ("Trichuris trichiura", ""),
    "tric": ("Trichuris trichiura", ""),
    "taenia": ("Taenia", "sp."),
    "tenia": ("Taenia", "sp."),
    "taeniasolium": ("Taenia solium", ""),
    "taeniasaginata": ("Taenia saginata", ""),
    "hymenolepisnana": ("Hymenolepis nana", ""),
    "hymenolepisdiminuta": ("Hymenolepis diminuta", ""),
    "hymenolepis": ("Hymenolepis", "sp."),
    "enterobius": ("Enterobius vermicularis", ""),
    "enterobiusvermicularis": ("Enterobius vermicularis", ""),
    "strongyloides": ("Strongyloides stercoralis", ""),
    "strongyloidesstercoralis": ("Strongyloides stercoralis", ""),
    "uncinaria": ("Ancylostomatidae", ""),
    "uncinarias": ("Ancylostomatidae", ""),
    "anquilostoma": ("Ancylostomatidae", ""),
    "ancylostoma": ("Ancylostoma", "sp."),
    "necator": ("Necator americanus", ""),
    "fasciola": ("Fasciola hepatica", ""),
    "giardia": ("Giardia intestinalis", ""),
    "giardi": ("Giardia intestinalis", ""),
    "giardialamblia": ("Giardia lamblia", ""),
    "giardiaintestinalis": ("Giardia intestinalis", ""),
    "giardiaduodenalis": ("Giardia duodenalis", ""),
    "entamoebacoli": ("Entamoeba coli", ""),
    "ecoli": ("Entamoeba coli", ""),
    "coli": ("Entamoeba coli", ""),
    "entamoebahistolytica": ("Entamoeba histolytica", ""),
    "histolytica": ("Entamoeba histolytica", ""),
    "entamoebahistolyticadispar": ("Entamoeba histolytica/dispar", ""),
    "entamoebahartmanni": ("Entamoeba hartmanni", ""),
    "hartmanni": ("Entamoeba hartmanni", ""),
    "hartma": ("Entamoeba hartmanni", ""),
    "hartm": ("Entamoeba hartmanni", ""),
    "endolimaxnana": ("Endolimax nana", ""),
    "endolimax": ("Endolimax nana", ""),
    "endnana": ("Endolimax nana", ""),
    "iodamoeba": ("Iodamoeba bütschlii", ""),
    "iodamoebabutschlii": ("Iodamoeba bütschlii", ""),
    "iodamoebabuetschlii": ("Iodamoeba bütschlii", ""),
    "ioda": ("Iodamoeba bütschlii", ""),
    "blastocystis": ("Blastocystis", "spp."),
    "blastocystishominis": ("Blastocystis hominis", ""),
    "blast": ("Blastocystis", "spp."),
    "chilomastix": ("Chilomastix mesnili", ""),
    "chilomastixmesnili": ("Chilomastix mesnili", ""),
    "chillom": ("Chilomastix mesnili", ""),
    "balantidium": ("Balantidium coli", ""),
    "balantidiumcoli": ("Balantidium coli", ""),
    "cryptosporidium": ("Cryptosporidium", "spp."),
    "cyclospora": ("Cyclospora cayetanensis", ""),
    "cystoisospora": ("Cystoisospora belli", ""),
    "isospora": ("Cystoisospora belli", ""),
}
_NON_PARASITE = {
    "negativo", "negative", "artefacto", "artefactos", "artifact", "artifacts", "fondo",
    "background", "detritus", "nada", "none", "otro", "otros", "other", "noparasito",
    "sinparasito", "levadura", "levaduras", "yeast",
}


def _norm(label: str) -> str:
    s = unicodedata.normalize("NFKD", str(label)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def sci_name(label: str):
    """(parte en cursiva, sufijo en redonda, es_parásito) para una clase del modelo."""
    key = _norm(label)
    if key in _NON_PARASITE:
        return "", str(label).replace("_", " ").strip().capitalize(), False
    if key in _SCI:
        it, suf = _SCI[key]
        return it, suf, True
    # Coincidencia por prefijo (p. ej. "ascaris_fertil" -> Ascaris lumbricoides)
    for k in sorted(_SCI, key=len, reverse=True):
        if len(k) >= 5 and key.startswith(k):
            it, suf = _SCI[k]
            return it, suf, True
    # Formato genérico: "Género especie", con sp./spp. en redonda
    words = re.sub(r"[_\-]+", " ", str(label)).split()
    if not words:
        return str(label), "", True
    suffix = ""
    if words[-1].lower().rstrip(".") in {"sp", "spp"}:
        suffix = words.pop().lower().rstrip(".") + "."
    words = [words[0].capitalize()] + [w.lower() for w in words[1:]]
    return " ".join(words), suffix, True


def plain_name(label: str) -> str:
    it, suf, _ = sci_name(label)
    return " ".join(p for p in (it, suf) if p)


def html_name(label: str) -> str:
    it, suf, is_par = sci_name(label)
    if not is_par:
        return suf
    return f"<i>{it}</i>" + (f" {suf}" if suf else "")


# --------------------------------------------------------------------------- #
# Fuentes
# --------------------------------------------------------------------------- #
def _font_dirs():
    dirs = ["staticFonts"]
    try:
        import matplotlib

        dirs.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf"))
    except Exception:
        pass
    dirs += ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/TTF"]
    return dirs


def _font(names, size):
    for d in _font_dirs():
        for n in names:
            path = os.path.join(d, n)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def fonts(size):
    return {
        "bold": _font(["DejaVuSans-Bold.ttf"], size),
        "italic": _font(["DejaVuSerif-Italic.ttf", "DejaVuSans-Oblique.ttf"], size),
        "regular": _font(["DejaVuSerif.ttf", "DejaVuSans.ttf"], size),
        "small": _font(["DejaVuSans.ttf"], int(size * 0.72)),
    }


def _text_w(draw, text, font):
    if not text:
        return 0
    b = draw.textbbox((0, 0), text, font=font)
    return b[2] - b[0]


def _logo():
    """Logo de ParasiTech recortado a su contenido (recuadro blanco con la marca)."""
    mark = Image.open(WATERMARK).convert("RGBA")
    bbox = mark.getchannel("A").point(lambda a: 255 if a > 10 else 0).getbbox()
    return mark.crop(bbox) if bbox else mark


# --------------------------------------------------------------------------- #
# Imagen de salida
# --------------------------------------------------------------------------- #
def render_report(image: Image.Image, items, parasite: str, min_conf: float) -> Image.Image:
    """Imagen a resolución completa con recuadros numerados y una banda negra de leyenda.

    items: [{"n", "full_box", "label", "confidence"}] ya filtrados para el reporte.
    """
    img = image.convert("RGB").copy()
    W, H = img.size
    draw = ImageDraw.Draw(img)

    fs = max(16, int(W * 0.017))          # números sobre la imagen
    F = fonts(fs)
    lw = max(3, W // 360)
    r_badge = int(fs * 0.8)

    # Recuadros con su número (sin nombre sobre la imagen)
    for it in items:
        x0, y0, x1, y1 = it["full_box"]
        draw.rectangle((x0, y0, x1, y1), outline=BOX_COLOR, width=lw)
        cx = min(max(r_badge, x0), W - r_badge)
        cy = y0 - r_badge if y0 - 2 * r_badge >= 0 else min(H - r_badge, y1 + r_badge)
        draw.ellipse((cx - r_badge, cy - r_badge, cx + r_badge, cy + r_badge),
                     fill=BOX_COLOR, outline=(255, 255, 255), width=max(2, lw // 2))
        draw.text((cx, cy), str(it["n"]), fill=TEXT, font=F["bold"], anchor="mm")

    # Logo de ParasiTech (una sola vez), dentro de la imagen en la esquina inferior derecha
    logo = _logo()
    wm_w = max(140, int(W * 0.15))
    wm = logo.resize((wm_w, int(logo.height * wm_w / logo.width)), Image.LANCZOS)
    margin = int(W * 0.015)
    img.paste(wm, (W - wm.width - margin, H - wm.height - margin), wm)

    # ---- Banda negra inferior: leyenda compacta en línea ---------------------
    ls = max(13, int(W * 0.0115))          # tamaño de letra de la leyenda
    L = fonts(ls)
    rb = int(ls * 0.72)                     # radio del número en la leyenda
    pad_x, pad_y = int(W * 0.025), int(ls * 0.9)
    gap = int(ls * 2.2)
    line_h = int(ls * 1.75)

    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    entries = []
    for it in items:
        ital, suf, _ = sci_name(it["label"])
        conf = f"{it['confidence']:.1f} %".replace(".", ",")
        if it["confidence"] < min_conf:
            conf += " (manual)"
        w_e = (2 * rb + int(ls * 0.45) + _text_w(tmp, ital, L["italic"])
               + (_text_w(tmp, " " + suf, L["regular"]) if suf else 0)
               + int(ls * 0.5) + _text_w(tmp, conf, L["small"]))
        entries.append((it["n"], ital, suf, conf, w_e))

    # Distribución en flujo: uno seguido del otro, salta de línea solo si no cabe
    max_w = W - 2 * pad_x
    lines, cur, cur_w = [], [], 0
    for e in entries:
        need = e[4] + (gap if cur else 0)
        if cur and cur_w + need > max_w:
            lines.append(cur)
            cur, cur_w = [], 0
            need = e[4]
        cur.append(e)
        cur_w += need
    if cur:
        lines.append(cur)

    head_h = int(ls * 1.6)
    body_h = max(1, len(lines)) * line_h
    disc_font = L["small"]
    disc_lines = _wrap(tmp, "⚠ " + DISCLAIMER, disc_font, max_w)
    disc_h = int(ls * 0.6) + len(disc_lines) * int(ls * 1.25)
    band_h = pad_y + head_h + int(ls * 0.5) + body_h + disc_h + pad_y
    out = Image.new("RGB", (W, H + band_h), BAND_BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)

    # Encabezado: datos del análisis a la izquierda, número de hallazgos a la derecha
    top = H + pad_y
    head = (f"{parasite.upper()}   ·   confianza ≥ {min_conf:.0f} %   ·   "
            f"{date.today().strftime('%d/%m/%Y')}")
    d.text((pad_x, top + head_h // 2), head, fill=MUTED, font=L["small"], anchor="lm")
    count = f"{len(entries)} hallazgo{'s' if len(entries) != 1 else ''}"
    d.text((W - pad_x, top + head_h // 2), count, fill=MUTED, font=L["small"], anchor="rm")
    sep_y = top + head_h + int(ls * 0.2)
    d.line((pad_x, sep_y, W - pad_x, sep_y), fill=(60, 60, 60), width=max(1, ls // 12))

    y = sep_y + int(ls * 0.3)
    if not entries:
        d.text((pad_x, y + line_h // 2), f"Sin hallazgos con confianza ≥ {min_conf:.0f} %",
               fill=TEXT, font=L["regular"], anchor="lm")
    for li, line in enumerate(lines):
        cy = y + li * line_h + line_h // 2
        x = pad_x
        for n, ital, suf, conf, w_e in line:
            d.ellipse((x, cy - rb, x + 2 * rb, cy + rb), fill=BOX_COLOR)
            d.text((x + rb, cy), str(n), fill=TEXT, font=L["bold"], anchor="mm")
            tx = x + 2 * rb + int(ls * 0.45)
            d.text((tx, cy), ital, fill=TEXT, font=L["italic"], anchor="lm")
            tx += _text_w(d, ital, L["italic"])
            if suf:
                d.text((tx, cy), " " + suf, fill=TEXT, font=L["regular"], anchor="lm")
                tx += _text_w(d, " " + suf, L["regular"])
            d.text((tx + int(ls * 0.5), cy), conf, fill=MUTED, font=L["small"], anchor="lm")
            x += w_e + gap

    # Aviso: herramienta de apoyo, requiere confirmación por personal experto
    dy = y + max(1, len(lines)) * line_h + int(ls * 0.6)
    for i, txt in enumerate(disc_lines):
        d.text((pad_x, dy + i * int(ls * 1.25)), txt, fill=(150, 150, 150), font=disc_font)
    return out


def _wrap(draw, text, font, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        test = f"{cur} {w}".strip()
        if cur and _text_w(draw, test, font) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = test
    if cur:
        lines.append(cur)
    return lines


def encode(img: Image.Image, fmt: str = "JPEG", max_side: int | None = None) -> bytes:
    im = img
    if max_side and max(img.size) > max_side:
        im = img.copy()
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = BytesIO()
    if fmt == "JPEG":
        im.save(buf, format="JPEG", quality=95, subsampling=0, optimize=True)
    else:
        im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
