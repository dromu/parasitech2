import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from prediction import predictParasite
from coordenates import calculate_rect_coords
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import shutil

TEMP_DIR = "temp_files"

def setup_temp_dir():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

def reset_temp_dir():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)

def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

def get_image_path(option, uploaded_file, img_dir, idm):
    if option == 'Subir' and uploaded_file:
        temp_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(temp_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())
        return temp_path
    
    elif option == "Galeria":
        reset_temp_dir()
        img_file_name = idm.get_image(st.session_state["image_index"])
        return os.path.join(img_dir, img_file_name)
    return "staticImages/parasitechLogo.png"

def display_predictions(preview_imgs, parasite, resized_img, coord):
    st.session_state.predecir = True
    columns = st.columns(3)
    img_bytes = None

    for idx, (preview_img, box_coords) in enumerate(zip(preview_imgs, coord)):
        preview_img[0].thumbnail((200, 200))
        col = columns[idx % 3]
        with col:
            col.image(preview_img[0])
            pred_class, _, probs = predictParasite(preview_img[0], parasite)
            max_prob = round(probs.max().item() * 100, 2)
            st.write(pred_class, f"{max_prob}%")

            draw = ImageDraw.Draw(resized_img)
            draw.rectangle(box_coords, outline="blue", width=3)

            font = ImageFont.truetype("DejaVuSans.ttf", 15) if True else ImageFont.load_default()
            x_center = ((box_coords[0] + box_coords[2]) // 2)-30
            draw.text((x_center, box_coords[3]), pred_class, fill="black", font=font)
            draw.text((x_center, box_coords[3] + 20), f"{max_prob}%", fill="black", font=font)

    overlay_image = Image.open("staticImages/parasitechWatermark.png").resize((189, 112))
    resized_img.paste(overlay_image, (resized_img.width - overlay_image.width, resized_img.height - overlay_image.height), overlay_image)

    img_bytes = BytesIO()
    resized_img.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    st.download_button("Descargar imagen", data=img_bytes, file_name="parasiTech.png", mime="image/png")

def run():
    img_dir = "imgExamples"
    st.sidebar.image("staticImages/parasitechLogo.png")
    parasite = st.sidebar.radio("Especie:", ["Protozoo", "Helminto"])
    slider_contour = st.sidebar.slider("Selección de umbral:", 0, 255, 90)
    opcion = st.sidebar.radio("Selecciona una opción:", ["Subir", "Galeria"])
    
    if opcion == "Subir":
        uploaded_file = st.sidebar.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
    
    elif opcion == "Galeria":
        st.sidebar.selectbox(
                "Files",
                st.session_state["files"],
                index=st.session_state["image_index"],
                on_change=go_to_image,
                key="file",
            )
        uploaded_file = None
    
    idm = ImageDirManager(img_dir)
    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])

    img_path = get_image_path(opcion, uploaded_file, img_dir, idm)
    
    if img_path == "staticImages/parasitechLogo.png":
        slider_contour = 120


    im = ImageManager(img_path, slider_contour)
    resized_img = im.resizing_img(max_height=1000, max_width=1000)
    rects = st_img_label(resized_img, box_color="blue", rects=im.get_resized_rects())

    if st.sidebar.button("Predecir") and rects:
        coord = calculate_rect_coords(rects)
        preview_imgs = im.init_annotation(rects)
        display_predictions(preview_imgs, parasite, resized_img, coord)

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Parasitech", page_icon="🖼️")
    setup_temp_dir()
    run()
