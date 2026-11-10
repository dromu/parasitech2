"""ParasiTech: apoyo al diagnóstico parasitológico con inteligencia artificial."""
import hashlib
import os
from datetime import date
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

from contorCircle import countourCircle
from prediction import get_learner, predict_topk
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
TEMP_DIR = "temp_files"
GALLERY_DIR = "imgExamples"
LOGO = "staticImages/parasitechLogo.png"
WATERMARK = "staticImages/parasitechWatermark.png"

MAX_PROCESS_SIDE = 2000      # lado máximo de la imagen antes de procesarla
CANVAS_MAX = (720, 960)      # alto y ancho máximos del lienzo de anotación
LOW_CONFIDENCE = 60.0        # por debajo de este % se marca para revisión
BOX_COLOR = "#E0115F"

SPECIES = {
    "Protozoo": {"model": "EfficientNet-B5", "icon": "🦠"},
    "Helminto": {"model": "ViT-Small · patch16 · 224", "icon": "🪱"},
}

st.set_page_config(
    page_title="ParasiTech · Diagnóstico parasitológico con IA",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 1rem;}
[data-testid="stSidebar"] [data-testid="stImage"] {margin-bottom: -0.5rem;}
.pt-hero {
    background: linear-gradient(120deg, #5B2A5E 0%, #7A3E7E 55%, #9C5A9F 100%);
    color: #fff; border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
}
.pt-hero h1 {color: #fff; font-size: 1.9rem; margin: 0 0 .3rem 0; padding: 0;}
.pt-hero p {margin: 0; opacity: .92; font-size: 1.02rem;}
.pt-chip {
    display: inline-block; padding: .18rem .65rem; border-radius: 999px; font-size: .8rem;
    font-weight: 600; margin: .6rem .4rem 0 0; background: rgba(255,255,255,.18); color: #fff;
}
.pt-step {
    border: 1px solid #E7E0EC; border-radius: 14px; padding: 1.1rem 1.2rem; height: 100%;
    background: #FBF9FC;
}
.pt-step .num {
    display: inline-flex; align-items: center; justify-content: center; width: 2rem;
    height: 2rem; border-radius: 50%; background: #5B2A5E; color: #fff; font-weight: 700;
    margin-bottom: .5rem;
}
.pt-step h4 {margin: .2rem 0 .3rem 0; padding: 0; font-size: 1.05rem;}
.pt-step p {margin: 0; color: #5d5566; font-size: .93rem;}
.pt-label {font-size: 1.15rem; font-weight: 700; margin: .25rem 0 .1rem 0; color: #1F1A24;}
.pt-region {font-size: .78rem; font-weight: 700; letter-spacing: .04em; color: #8a7f93;
    text-transform: uppercase;}
.pt-badge-ok, .pt-badge-warn {
    display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem;
    font-weight: 700;
}
.pt-badge-ok {background: #E6F4EA; color: #1E6B35;}
.pt-badge-warn {background: #FFF4E5; color: #8A4B00;}
.pt-alt {font-size: .85rem; color: #5d5566; margin: .1rem 0;}
.pt-footer {
    margin-top: 2.5rem; padding: 1.1rem 0 .4rem 0; border-top: 1px solid #E7E0EC;
    text-align: center; font-size: .82rem; color: #6b6275; line-height: 1.6;
}
.pt-footer a {color: #5B2A5E; text-decoration: none; font-weight: 600;}
</style>
"""


# --------------------------------------------------------------------------- #
# Utilidades de imagen
# --------------------------------------------------------------------------- #
@st.cache_data(max_entries=8, show_spinner=False)
def processed_image(path: str, threshold: int) -> Image.Image:
    """Abre, orienta, reduce y recorta el campo del microscopio (cacheado)."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((MAX_PROCESS_SIDE, MAX_PROCESS_SIDE))
    return countourCircle(img, threshold)


class CachedImageManager(ImageManager):
    def initProcess(self):
        self._img = processed_image(self._filename, self.umbral)


def save_upload(uploaded) -> str:
    """Guarda la imagen subida con un nombre único por contenido."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    data = uploaded.getvalue()
    ext = os.path.splitext(uploaded.name)[1].lower() or ".png"
    path = os.path.join(TEMP_DIR, hashlib.md5(data).hexdigest() + ext)
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(data)
    return path


def gallery_files():
    exts = (".png", ".jpg", ".jpeg")
    return sorted(f for f in os.listdir(GALLERY_DIR) if f.lower().endswith(exts))


def get_font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def rect_to_box(rect):
    left, top = int(rect["left"]), int(rect["top"])
    return left, top, left + int(rect["width"]), top + int(rect["height"])


def annotate(image: Image.Image, results) -> bytes:
    """Dibuja los recuadros numerados con su clase y agrega la marca de agua."""
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    font = get_font(max(13, img.width // 55))

    for r in results:
        x0, y0, x1, y1 = r["box"]
        draw.rectangle((x0, y0, x1, y1), outline=BOX_COLOR, width=3)
        text = f"{r['n']}. {r['label']}  {r['confidence']:.1f}%"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y0 - th - 8 if y0 - th - 8 >= 0 else y1 + 2
        tx = min(max(0, x0), max(0, img.width - tw - 8))
        draw.rectangle((tx, ty, tx + tw + 8, ty + th + 6), fill=BOX_COLOR)
        draw.text((tx + 4, ty + 2 - tb[1]), text, fill="white", font=font)

    try:
        mark = Image.open(WATERMARK).convert("RGBA")
        w = max(120, img.width // 5)
        mark = mark.resize((w, int(mark.height * w / mark.width)))
        img.paste(mark, (img.width - mark.width - 8, img.height - mark.height - 8), mark)
    except OSError:
        pass

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Secciones de la interfaz
# --------------------------------------------------------------------------- #
def sidebar():
    st.sidebar.image(LOGO, use_column_width=True)

    st.sidebar.markdown("##### 1 · Tipo de parásito")
    parasite = st.sidebar.radio(
        "Tipo de parásito",
        list(SPECIES),
        captions=[SPECIES[s]["model"] for s in SPECIES],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("##### 2 · Imagen de la muestra")
    source = st.sidebar.radio(
        "Origen", ["Subir imagen", "Galería de ejemplos"], label_visibility="collapsed"
    )
    img_path, img_name = None, None
    if source == "Subir imagen":
        uploaded = st.sidebar.file_uploader(
            "Imagen de microscopio", type=["png", "jpg", "jpeg"],
            help="Fotografía tomada a través del microscopio (PNG o JPG, máx. 25 MB).",
        )
        if uploaded:
            img_path, img_name = save_upload(uploaded), uploaded.name
    else:
        files = gallery_files()
        choice = st.sidebar.selectbox(
            "Ejemplo", files, format_func=lambda f: os.path.splitext(f)[0]
        )
        if choice:
            img_path, img_name = os.path.join(GALLERY_DIR, choice), choice

    with st.sidebar.expander("⚙️ Ajustes de imagen"):
        threshold = st.slider(
            "Umbral de recorte", 0, 255, 90,
            help="Separa el campo circular del microscopio del fondo oscuro. "
                 "Súbelo si el recorte incluye bordes negros; bájalo si corta parte de la muestra.",
        )

    st.sidebar.markdown("##### 3 · Análisis")
    analyze = st.sidebar.button(
        "🔍  Analizar regiones", type="primary", use_container_width=True,
        disabled=img_path is None,
    )
    if st.sidebar.button("Limpiar resultados", use_container_width=True):
        st.session_state.pop("results", None)

    st.sidebar.caption(
        "Herramienta de apoyo. Los resultados deben ser confirmados por personal "
        "de laboratorio calificado."
    )
    return parasite, img_path, img_name, threshold, analyze


def header(parasite):
    info = SPECIES[parasite]
    st.markdown(
        f"""
        <div class="pt-hero">
          <h1>Análisis parasitológico asistido por IA</h1>
          <p>Marca las estructuras sospechosas en la imagen del microscopio y obtén
             una clasificación automática con su nivel de confianza.</p>
          <span class="pt-chip">{info['icon']} {parasite}</span>
          <span class="pt-chip">🧠 {info['model']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state():
    steps = [
        ("1", "Elige el tipo", "Selecciona si buscas <b>protozoos</b> o <b>helmintos</b>; "
                               "cada uno usa un modelo especializado."),
        ("2", "Carga la imagen", "Sube una foto del microscopio o prueba con una imagen "
                                 "de la <b>galería de ejemplos</b>."),
        ("3", "Marca y analiza", "Dibuja un recuadro sobre cada estructura y pulsa "
                                 "<b>Analizar regiones</b>."),
    ]
    cols = st.columns(3)
    for col, (n, title, text) in zip(cols, steps):
        col.markdown(
            f'<div class="pt-step"><div class="num">{n}</div><h4>{title}</h4><p>{text}</p></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.info("👈 Comienza cargando una imagen desde el panel lateral.")


def clamp_rects(rects, width, height):
    """Ajusta los recuadros a los límites de la imagen y descarta los vacíos."""
    out = []
    for r in rects:
        x0, y0 = max(0.0, r["left"]), max(0.0, r["top"])
        x1 = min(float(width), r["left"] + r["width"])
        y1 = min(float(height), r["top"] + r["height"])
        if x1 - x0 >= 4 and y1 - y0 >= 4:
            out.append({**r, "left": x0, "top": y0, "width": x1 - x0, "height": y1 - y0})
    return out


def run_analysis(im, rects, parasite, resized_img, sig):
    rects = clamp_rects(rects, resized_img.width, resized_img.height)
    if not rects:
        st.warning("Los recuadros están fuera de la imagen. Muévelos sobre la muestra.", icon="✏️")
        return
    boxes = [rect_to_box(r) for r in rects]
    crops = [c for c, _ in im.init_annotation([dict(r) for r in rects])]

    results = []
    progress = st.progress(0.0, text="Analizando regiones…")
    for i, (crop, box) in enumerate(zip(crops, boxes), start=1):
        top = predict_topk(crop, parasite, k=3)
        label, prob = top[0]
        results.append({
            "n": i, "box": box, "crop": crop, "label": label,
            "confidence": round(prob * 100, 2),
            "alternatives": [(l, round(p * 100, 2)) for l, p in top[1:]],
        })
        progress.progress(i / len(rects), text=f"Analizando región {i} de {len(rects)}…")
    progress.empty()

    st.session_state["results"] = {
        "sig": sig, "parasite": parasite, "items": results,
        "annotated": annotate(resized_img, results),
    }


def show_results(res, img_name, stale):
    items = res["items"]
    st.subheader("Resultados")
    if stale:
        st.warning(
            "Los recuadros o los ajustes cambiaron desde el último análisis. "
            "Pulsa **Analizar regiones** para actualizar los resultados.", icon="⚠️",
        )

    confs = [r["confidence"] for r in items]
    low = sum(c < LOW_CONFIDENCE for c in confs)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Regiones analizadas", len(items))
    m2.metric("Confianza promedio", f"{sum(confs) / len(confs):.1f}%")
    m3.metric("Requieren revisión", low, help=f"Regiones con confianza menor a {LOW_CONFIDENCE:.0f}%.")
    m4.metric("Tipo de parásito", res["parasite"], help=f"Modelo: {SPECIES[res['parasite']]['model']}")

    cols = st.columns(3)
    for r in items:
        with cols[(r["n"] - 1) % 3].container(border=True):
            c_img, c_txt = st.columns([1, 1.6], vertical_alignment="center")
            c_img.image(r["crop"], use_column_width=True)
            with c_txt:
                badge = (
                    '<span class="pt-badge-ok">Confianza alta</span>'
                    if r["confidence"] >= LOW_CONFIDENCE
                    else '<span class="pt-badge-warn">Revisar</span>'
                )
                st.markdown(
                    f'<div class="pt-region">Región {r["n"]}</div>'
                    f'<div class="pt-label">{r["label"]}</div>{badge}',
                    unsafe_allow_html=True,
                )
            st.progress(min(r["confidence"] / 100, 1.0), text=f"Confianza: {r['confidence']:.1f}%")
            if r["alternatives"]:
                alts = " · ".join(f"{l} ({p:.1f}%)" for l, p in r["alternatives"])
                st.markdown(f'<div class="pt-alt">Otras opciones: {alts}</div>', unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([1.5, 1])
    with left:
        st.markdown("**Imagen anotada**")
        st.image(res["annotated"], use_column_width=True)
    with right:
        st.markdown("**Resumen**")
        df = pd.DataFrame([
            {
                "Región": r["n"],
                "Clasificación": r["label"],
                "Confianza (%)": r["confidence"],
                "Estado": "OK" if r["confidence"] >= LOW_CONFIDENCE else "Revisar",
            }
            for r in items
        ])
        st.dataframe(df, hide_index=True, use_container_width=True)

        base = os.path.splitext(img_name or "muestra")[0]
        st.download_button(
            "⬇️  Descargar imagen anotada", data=res["annotated"],
            file_name=f"parasitech_{base}.png", mime="image/png", use_container_width=True,
        )
        csv = df.assign(Tipo=res["parasite"], Imagen=img_name).to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️  Descargar resultados (CSV)", data=csv,
            file_name=f"parasitech_{base}.csv", mime="text/csv", use_container_width=True,
        )


def footer():
    year = date.today().year
    st.markdown(
        f"""
        <div class="pt-footer">
          © 2024–{year} ParasiTech · Todos los derechos reservados<br>
          Desarrollado por
          <a href="https://orcid.org/0009-0001-9428-4192" target="_blank">Jader Muñoz</a>,
          <a href="https://orcid.org/0000-0003-3538-6313" target="_blank">Reinel Vasquez</a> y
          <a href="https://orcid.org/0000-0003-1548-942X" target="_blank">Rubiel Vargas</a><br>
          Financiado por la «Implementación del Proyecto Jóvenes Investigadores e Innovadores
          en el Departamento del Cauca» BPIN 2020000100043
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Aplicación
# --------------------------------------------------------------------------- #
def main():
    st.markdown(CSS, unsafe_allow_html=True)
    parasite, img_path, img_name, threshold, analyze = sidebar()
    header(parasite)

    if img_path is None:
        empty_state()
        footer()
        return

    try:
        im = CachedImageManager(img_path, threshold)
    except Exception:
        st.error("No se pudo abrir la imagen. Verifica que sea un archivo PNG o JPG válido.")
        footer()
        return

    resized_img = im.resizing_img(*CANVAS_MAX)

    st.markdown("**Marca las regiones a analizar**")
    st.caption(
        "Pulsa **+ Agregar recuadro**, arrástralo sobre cada estructura y ajusta su tamaño "
        "desde las esquinas. Puedes marcar varias regiones a la vez."
    )
    rects = st_img_label(resized_img, box_color=BOX_COLOR, rects=im.get_resized_rects())

    sig = (img_path, threshold, parasite, tuple(rect_to_box(r) for r in rects))

    if analyze:
        if not rects:
            st.warning("Agrega al menos un recuadro sobre la imagen antes de analizar.", icon="✏️")
        else:
            try:
                with st.spinner(f"Cargando el modelo de {parasite}…"):
                    get_learner(parasite)
            except Exception as e:
                print(f"[ParasiTech] Error cargando modelo {parasite}: {e}")
                st.error(f"El modelo de {parasite} no está disponible en este momento. "
                         "Intenta de nuevo en unos minutos.")
            else:
                run_analysis(im, rects, parasite, resized_img, sig)

    res = st.session_state.get("results")
    if res and res["sig"][0] == img_path:
        st.divider()
        show_results(res, img_name, stale=res["sig"] != sig)

    footer()


main()
