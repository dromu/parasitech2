import streamlit as st
import os
from streamlit_img_label import st_img_label
from streamlit_img_label.manage import ImageManager, ImageDirManager
from prediction import predictParasite
import os 
import shutil
from coordenates import *
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

def run(img_dir, labels):

    temp_dir = "temp_files"
    os.makedirs(temp_dir, exist_ok=True) 

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

    opciones = ['Subir', 'Galeria']
    opcion_seleccionada = st.sidebar.radio('Selecciona una opción:', opciones)

    
    if opcion_seleccionada == 'Subir':
        
        uploaded_file = st.sidebar.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])
        if uploaded_file:            
            os.makedirs(temp_dir, exist_ok=True) 

            temp_path = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_path, "wb") as temp_file:
                temp_file.write(uploaded_file.read())

            st.write("Ruta del archivo temporal:", temp_path)
            img_path = temp_path
        
        else: 
            img_path = "parasitechLogo.png"

    elif opcion_seleccionada == "Galeria":
        
        #Elimina la carpeta residual si existe 

        if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
            

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

            coord = calculate_rect_coords(rects)
            
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

                                draw = ImageDraw.Draw(resized_img)
                                box_coords = coord[idx_imagen]
                                x1, y1,x2,y2 = box_coords
                                draw.rectangle(box_coords, outline="blue", width=3)

                                try:
                                    font = ImageFont.truetype("DejaVuSans.ttf", 15) # Carga una fuente (puedes cambiarla)
                                except IOError:
                                    font = ImageFont.load_default()

                                x0 = int((x2+x1)/2)-30
                                

                                text_position = (x0,y2)  # Coordenadas (x, y)
                                draw.text(text_position, pred_class, fill="black", font=font)

                                text_position = (x0,y2+20) # Coordenadas
                                draw.text(text_position, str(maxPor)+" %", fill="black", font=font)

                                overlay_image = Image.open("parasitechResultado.png")
                                overlay_image = overlay_image.resize((189, 112)) 
                                overlay_width, overlay_height = overlay_image.size

                                base_width, base_height = resized_img.size
                                position = (base_width - overlay_width, base_height - overlay_height)

                                resized_img.paste(overlay_image, position, overlay_image)

                                # Guardar la imagen en un archivo temporal en memoria
                                img_bytes = BytesIO()
                                resized_img.save(img_bytes, format="PNG")
                                img_bytes.seek(0) 

                                st.download_button(
                                label="Descargar imagen",
                                data=img_bytes,
                                file_name="Parasitech.png",
                                mime="image/png"
                                )
                                
                          
                



                        
            st.session_state.predecir = False


            

if __name__ == "__main__":
    custom_labels = ["", "dog", "cat"]

    st.set_page_config(layout="wide", page_title="Parasitech", page_icon="🖼️")

    run("img_dir", custom_labels)
