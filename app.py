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
    
    def refresh():
        st.session_state["files"] = idm.get_all_files()
        st.session_state["annotation_files"] = idm.get_exist_annotation_files()
        st.session_state["image_index"] = 0

    def next_image():
        image_index = st.session_state["image_index"]
        if image_index < len(st.session_state["files"]) - 1:
            st.session_state["image_index"] += 1
        else:
            st.warning("This is the last image.")

    def previous_image():
        image_index = st.session_state["image_index"]
        if image_index > 0:
            st.session_state["image_index"] -= 1
        else:
            st.warning("This is the first image.")

    def next_annotate_file():
        image_index = st.session_state["image_index"]
        next_image_index = idm.get_next_annotation_image(image_index)
        if next_image_index is not None:
            st.session_state["image_index"] = next_image_index
        else:
            st.warning("All images are annotated.")
            next_image()

    def go_to_image():
        file_index = st.session_state["files"].index(st.session_state["file"])
        st.session_state["image_index"] = file_index

    # Sidebar: show status
    n_files = len(st.session_state["files"])
    n_annotate_files = len(st.session_state["annotation_files"])

    st.sidebar.write("Total files:", n_files)
    st.sidebar.write("Total annotated files:", n_annotate_files)
    st.sidebar.write("Remaining files:", n_files - n_annotate_files)

    parasite = st.sidebar.radio("Especie:", ("Protozoo", "Helminto"))

    sliderContour = st.sidebar.slider("Selección de umbral:",0,255,90)

    st.sidebar.selectbox(
        "Files",
        st.session_state["files"],
        index=st.session_state["image_index"],
        on_change=go_to_image,
        key="file",
    )


    # Botons 
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.button(label="Previous image", on_click=previous_image)
    with col2:
        st.button(label="Next image", on_click=next_image)
    st.sidebar.button(label="Next need annotate", on_click=next_annotate_file)
    st.sidebar.button(label="Refresh", on_click=refresh)

    # Main content: annotate images
    img_file_name   = idm.get_image(st.session_state["image_index"])
    img_path        = os.path.join(img_dir, img_file_name)
    im              = ImageManager(img_path, sliderContour)
    
    resized_img     = im.resizing_img(max_height=1000,max_width=1000)
    resized_rects   = im.get_resized_rects()
    rects           = st_img_label(resized_img, box_color="blue", rects=resized_rects)

    def annotate():
        im.save_annotation()
        image_annotate_file_name = img_file_name.split(".")[0] + ".xml"
        if image_annotate_file_name not in st.session_state["annotation_files"]:
            st.session_state["annotation_files"].append(image_annotate_file_name)
        next_annotate_file()



    if rects:
        st.button(label="Save", on_click=annotate)
        preview_imgs = im.init_annotation(rects)

        for i, prev_img in enumerate(preview_imgs):

            
            prev_img[0].thumbnail((400, 400))

            col1, col2 = st.columns(2)
            with col1:

                col1.image(prev_img[0])
                pred_class,pred_idx,probs = predictParasite(prev_img[0],parasite)
                maxPor= round(probs.max().item() * 100,2)
                st.write(pred_class,maxPor)

            with col2:
                default_index = 0
                if prev_img[1]:
                    default_index = labels.index(prev_img[1])

                st.write("hola mundo")

                select_label = col2.selectbox(
                    "Label", labels, key=f"label_{i}", index=default_index
                )
                im.set_annotation(i, select_label)

if __name__ == "__main__":
    custom_labels = ["", "dog", "cat"]

    st.set_page_config(layout="wide", page_title="Visualización de imágenes", page_icon="🖼️")
    uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

    # print(uploaded_file)

    # run(uploaded_file, custom_labels)


    run("img_dir", custom_labels)
