# rectangles = [
#     {'left': 150, 'top': 112.5, 'width': 201, 'height': 151, 'label': ''},
#     {'left': 153, 'top': 115.5, 'width': 201, 'height': 151, 'label': ''},
#     {'left': 156, 'top': 118.5, 'width': 201, 'height': 151, 'label': ''}
# ]

# Función para calcular las coordenadas iniciales y finales
def calculate_rect_coords(rectangles):
    rect_coords = []
    
    for rect in rectangles:
        left = int(rect['left'])
        top = int(rect['top'])
        right = int(left + rect['width'])  # Punto final en X
        bottom = int(top + rect['height'] ) # Punto final en Y
        rect_coords.append( (left, top, right, bottom) )
        
    return rect_coords

# # Llamamos a la función
# rect_coords = calculate_rect_coords(rectangles)

# # Imprimimos los resultados
# for idx, rect in enumerate(rect_coords):
#     print(f"Rectángulo {idx}:")
#     print(f"  Esquina superior izquierda: {rect['top_left']}")
#     print(f"  Esquina inferior derecha: {rect['bottom_right']}")
