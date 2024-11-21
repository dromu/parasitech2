import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from prediction import predictParasite
from coordTransf import transfCoord
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import io
from contorCircle import countourCircle


if 'condDial' not in st.session_state:
    st.session_state.condDial = True
if 'accepted' not in st.session_state:
    st.session_state.accepted = False  # Controlar si se aceptaron los términos y se completó la información

# Inicializar session_state si no está definido
if 'accepted' not in st.session_state:
    st.session_state.accepted = False
if 'condDial' not in st.session_state:
    st.session_state.condDial = True

@st.dialog("Notificación", width="large")  # Decorador para el diálogo
def vote():
    if not st.session_state.accepted:  # Mostrar solo si aún no se ha aceptado
        st.write("Al utilizar esta aplicación, aceptas que la información y los datos que suministres podrán ser utilizados para mejorar la funcionalidad y el rendimiento del sistema. Nos comprometemos a proteger tu privacidad y a utilizar tus datos de manera ética y responsable, de acuerdo con las leyes de protección de datos vigentes.")
        
        # Checkbox por defecto en True
        acepto = st.checkbox("Acepto los términos", value=True)
        
        # Botón Aceptar
        if st.button("Aceptar"):
            if acepto:  # Verifica que el usuario acepte los términos
                st.session_state.accepted = True  # Cambiar el estado a aceptado
                st.session_state.condDial = False  # Cierra el diálogo
                st.success("Gracias por aceptar los términos. Ahora puedes continuar.")  # Mensaje de éxito
            else:
                st.error("Debes aceptar los términos para continuar.")
    else:
        st.success("Gracias por aceptar los términos.")  # Mostrar mensaje de éxito si ya se aceptó

# Ejecutar el diálogo solo si es la primera vez o aún no se ha aceptado
if st.session_state.condDial:
    vote()  # Llama a la función para mostrar el contenido del diálogo


# Logo 
st.sidebar.image("parasitechLogo.png")

# st.sidebar.warning("Esta aplicación aún se encuentra en desarrollo. Algunas funcionalidades pueden no estar disponibles.")

# Crear la barra lateral
st.sidebar.header("Configuración")





# Specify canvas parameters in application
drawing_mode = st.sidebar.radio(
    "Drawing tool:", ("rect", "transform"))

parasite = st.sidebar.radio(
    "Especie:", ("Protozoo", "Helminto"))

stroke_width = 2

if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

sliderUmbral = st.sidebar.slider("Umbral de prediccion %",60,100,90)
sliderContour = st.sidebar.slider("Selección de umbral:",0,255,90)
okPredict = st.sidebar.button("Identificar")
#Button predecir



stroke_color = "000000"
bg_color = "#EEEEEE"
bg_image = st.file_uploader("Background image:", type=["png", "jpg"])


if bg_image != None:
    
    img = Image.open(bg_image)
    img = img.resize((2100, 2100), Image.LANCZOS)
    img = countourCircle(img, sliderContour)
    
    realtime_update = True


    

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=img,
        update_streamlit=realtime_update, 
        height= 750,
        width=750,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
        key="canvas",
        display_toolbar = True
    )



    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")
            
        # st.dataframe(objects)

    def generar_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    if okPredict: 

        item = True

        try : 
            nPred = []
            nPorc = []
            coordOut = []
            especie = {}
        
            df = pd.DataFrame()

            cols = ["left","top","width","height"]
            coordinates = objects[cols]

            info_message = st.info("Trabajando, un momento por favor ")

            for i in range(0,len(coordinates)):

                coord = coordinates.iloc[i].to_list() 
                coordOut.append(transfCoord(coord))
            


                pred_class,pred_idx,probs = predictParasite(parasite=parasite,
                                                        image= img,
                                                        coordinates=coord)
                
                


                maxPor= round(probs.max().item() * 100,2)

                if maxPor >=sliderUmbral:
                    nPred.append(pred_class)
                    nPorc.append(maxPor)

                else: 
                    nPred.append("No identificado")
                    nPorc.append("--")
                    



                
            info_message.empty()

            df["Prediccion"] = nPred
            df["Porcentaje"] = nPorc
            df["Coordenadas"] = coordOut
            

            st.dataframe(df)

            img = img.resize((750, 750), Image.LANCZOS)
            draw = ImageDraw.Draw(img)




            for i in range(len(df)):
                # Modificar la imagen de fondo para agregar texto
                if bg_image is not None:

                    # Definir la fuente (puedes usar una fuente específica si la deseas)
                    font_size = 13

                    try:
                        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                    except IOError:
                        font = ImageFont.load_default()

                    # Coordenadas para el texto
                    x0, x1,y0,y1 =  coordOut[i]
                    bbox = (x0,y0,x1,y1)

                    
                    text = nPred[i]  # Cambia el texto según lo que necesites
                    textPor = nPorc[i]

                    if text not in especie:
                        color = generar_color()
                        especie[text] =color
                    
                    text2 = nPred[i] + "\n  " + str(textPor) + "%" 
                    # Agregar texto a la imagen
                    draw.text((x0-15, y1+5), text2, fill="black", font=font)

                    draw.rectangle(bbox, outline=especie[text], width=3)


            # Mostrar la imagen modificada en Streamlit
            imgResult = st.image(img, caption="Reconocimiento de parasitos", use_column_width=True, )


            nPred = []
            nPorc = []
            coordOut = []
            especie = {}


            buf = io.BytesIO()
            img.save(buf, format='JPEG')  # Guarda la imagen en formato JPEG en el buffer
            # buf.seek(0)  # Regresa al inicio del buff

            downButton = st.download_button(label="Descargar imagen",
                data=buf,
                file_name='parasiTech_prediction.jpg',  # Nombre del archivo que se descargará
                mime='image/jpeg' ) # Tipo MIME de la imagen)
            

        except Exception as e:
            
            st.error("No hay una selección en la imagen de referencia ")


# Agregar un pie de página
footer = """
    <style>
        .footer {
            position: relative ;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: lightgray;
            text-align: center;
            padding: 0px;
        }
    </style>
    <div class="footer">
        <p>© 2024 ParasiTech - Todos los derechos reservados.<br>
        Desarrollado por 
        <a href="https://orcid.org/0009-0001-9428-4192" target="_blank">Jader Muñoz</a>, 
        <a href="https://orcid.org/0000-0003-3538-6313" target="_blank">Reinel Vasquez</a>, y
        <a href="https://orcid.org/0000-0003-1548-942X" target="_blank">Rubiel Vargas</a><br>
        Financiado por la "Implementación del Proyecto Jóvenes Investigadores e Innovadores en el Departamento del Cauca" BPIN 2020000100043</p>
    </div>
    """
st.markdown(footer, unsafe_allow_html=True)




# 