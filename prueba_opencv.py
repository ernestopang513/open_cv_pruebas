import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread('cuadro.png')
if imagen is None:
    print("Error: No se puede abrir o leer el archivo de imagen. Verifica la ruta y el nombre del archivo.")
else:
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplica un umbral para convertir la imagen a binaria
    _, th = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Encuentra contornos en la imagen binaria
    contornos, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print('hierarchy=', hierarchy)
    
    # Dibuja los contornos que cumplen con el criterio de área
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        if area > 3000:
            cv2.drawContours(imagen, [contorno], -1, (0, 255, 0), 3)
    
    # Muestra la imagen en escala de grises
    cv2.imshow('Imagen en escala de grises', gray)
    
    # Muestra la imagen binaria
    cv2.imshow('Imagen binaria', th)
    
    # Muestra la imagen original con todos los contornos dibujados
    cv2.imshow('Imagen con contornos', imagen)
    
    # Espera hasta que se presione una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()
