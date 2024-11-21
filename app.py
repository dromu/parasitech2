import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from prediction import predictParasite

def run(img_dir, labels):

    st.sidebar.image("parasitechLogo.png")
    # Eliminar configuración obsoleta
    idm = ImageDirManager(img_dir)

    if "files" not in st.session_state:
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0
    else:
        idm.set_all_files(st.session_state["files"])
        idm.set_annotation_files(st.session_state["annotation_files"])
    
    

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index
    
    parasite = st.sidebar.radio("Especie:", ("Protozoo", "Helminto"))
    sliderContour = st.sidebar.slider("Selección de umbral:",0,255,90)
    sliderUmbral = st.sidebar.slider("Umbral de prediccion %",60,100,90)

    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )




    # Main content: annotate images
    img_file_name   = idm.get_image(st.session_state["image_index"])
    img_path        = os.path.join(img_dir, img_file_name)
    im              = ImageManager(img_path, sliderContour)
    
    resized_img     = im.resizing_img(max_height=1000,max_width=1000)
    resized_rects   = im.get_resized_rects()
    rects           = st_img_label(resized_img, box_color="blue", rects=resized_rects)

    if "predecir" not in st.session_state:
        st.session_state.predecir = False
    

    if st.sidebar.button("Predecir"):
        st.session_state.predecir = True

    if rects:
        preview_imgs = im.init_annotation(rects)

        # Número de columnas
        columns = 3
        cantidad = len(preview_imgs)
        filas = cantidad // columns
        if cantidad % columns != 0:
            filas += 1

        # Crear las columnas una sola vez
        col1, col2, col3 = st.columns(columns)

        # Iterar sobre las filas
        for fila in range(filas):
            # Asignar las imágenes a las columnas en cada fila
            for col_idx, column in enumerate([col1, col2, col3]):
                # Calcular el índice de la imagen correspondiente
                idx_imagen = fila * columns + col_idx
                if idx_imagen < cantidad:
                    with column:
                        preview_imgs[idx_imagen][0].thumbnail((200, 200))  # Redimensionar la imagen
                        column.image(preview_imgs[idx_imagen][0])
                    

           
                        if st.session_state.predecir:
                                pred_class,pred_idx,probs = predictParasite(preview_imgs[idx_imagen][0],parasite)
                                maxPor= round(probs.max().item() * 100,2)
                                st.write(pred_class,maxPor)
                    
        st.session_state.predecir = False

if __name__ == "__main__":
    custom_labels = ["", "dog", "cat"]

    st.set_page_config(layout="wide", page_title="Parasitech", page_icon="🖼️")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

    # print(uploaded_file)

    # run(uploaded_file, custom_labels)

    run("img_dir", custom_labels)
