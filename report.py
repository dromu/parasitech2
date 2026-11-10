"""Nombres científicos e imagen de salida (reporte) de ParasiTech."""
import math
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

    fs = max(18, int(W * 0.021))
    F = fonts(fs)
    lw = max(3, W // 330)
    r_badge = int(fs * 0.85)

    # Recuadros con su número (sin nombre sobre la imagen)
    for it in items:
        x0, y0, x1, y1 = it["full_box"]
        draw.rectangle((x0, y0, x1, y1), outline=BOX_COLOR, width=lw)
        cx = min(max(r_badge, x0), W - r_badge)
        cy = y0 - r_badge if y0 - 2 * r_badge >= 0 else min(H - r_badge, y1 + r_badge)
        draw.ellipse((cx - r_badge, cy - r_badge, cx + r_badge, cy + r_badge),
                     fill=BOX_COLOR, outline=(255, 255, 255), width=max(2, lw // 2))
        draw.text((cx, cy), str(it["n"]), fill=TEXT, font=F["bold"], anchor="mm")

    # Marca de agua sobre la imagen (esquina inferior derecha)
    logo = _logo()
    wm_w = max(140, int(W * 0.16))
    wm = logo.resize((wm_w, int(logo.height * wm_w / logo.width)), Image.LANCZOS)
    alpha = wm.getchannel("A").point(lambda a: int(a * 0.85))
    wm.putalpha(alpha)
    pad_img = int(W * 0.015)
    img.paste(wm, (W - wm.width - pad_img, H - wm.height - pad_img), wm)

    # ---- Banda negra inferior con la leyenda --------------------------------
    pad = int(W * 0.03)
    gap = int(fs * 1.6)
    row_h = int(fs * 2.0)
    logo_h = int(fs * 3.4)
    band_logo = logo.resize((int(logo.width * logo_h / logo.height), logo_h), Image.LANCZOS)

    tmp = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    entries = []
    for it in items:
        ital, suf, is_par = sci_name(it["label"])
        conf = f"{it['confidence']:.1f} %".replace(".", ",")
        if it["confidence"] < min_conf:
            conf += " · incluida manualmente"
        w_e = (2 * r_badge + int(fs * 0.6) + _text_w(tmp, ital, F["italic"])
               + (_text_w(tmp, " " + suf, F["regular"]) if suf else 0)
               + _text_w(tmp, "   " + conf, F["small"]))
        entries.append((it["n"], ital, suf, conf, w_e))

    avail = W - 2 * pad - band_logo.width - gap
    col_w = max([e[4] for e in entries] + [1]) + gap
    ncols = max(1, min(len(entries) or 1, avail // col_w))
    nrows = math.ceil(len(entries) / ncols) if entries else 1

    title_h = int(fs * 1.9)
    content_h = title_h + nrows * row_h
    band_h = max(content_h, logo_h) + 2 * pad
    out = Image.new("RGB", (W, H + band_h), BAND_BG)
    out.paste(img, (0, 0))
    d = ImageDraw.Draw(out)

    top = H + pad
    title = (f"Hallazgos · {parasite} · confianza ≥ {min_conf:.0f} %"
             f" · {date.today().strftime('%d/%m/%Y')}")
    d.text((pad, top), title, fill=MUTED, font=F["small"])

    y = top + title_h
    if not entries:
        d.text((pad, y + row_h // 2), f"Sin hallazgos con confianza ≥ {min_conf:.0f} %",
               fill=TEXT, font=F["regular"], anchor="lm")
    for idx, (n, ital, suf, conf, _) in enumerate(entries):
        col, row = idx // nrows, idx % nrows
        x = pad + col * col_w
        cy = y + row * row_h + row_h // 2
        d.ellipse((x, cy - r_badge, x + 2 * r_badge, cy + r_badge), fill=BOX_COLOR)
        d.text((x + r_badge, cy), str(n), fill=TEXT, font=F["bold"], anchor="mm")
        tx = x + 2 * r_badge + int(fs * 0.6)
        d.text((tx, cy), ital, fill=TEXT, font=F["italic"], anchor="lm")
        tx += _text_w(d, ital, F["italic"])
        if suf:
            d.text((tx, cy), " " + suf, fill=TEXT, font=F["regular"], anchor="lm")
            tx += _text_w(d, " " + suf, F["regular"])
        d.text((tx, cy), "   " + conf, fill=MUTED, font=F["small"], anchor="lm")

    # Logo de la marca en la banda (derecha, centrado verticalmente)
    lx = W - pad - band_logo.width
    ly = H + (band_h - band_logo.height) // 2
    out.paste(band_logo, (lx, ly), band_logo)
    return out


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
