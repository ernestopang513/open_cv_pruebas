import cv2
import numpy as np

# Lee la imagen
imagen = cv2.imread('cuadro.png')
if imagen is None:
    print("Error: No se puede abrir o leer el archivo de imagen. Verifica la ruta y el nombre del archivo.")
else:
    # Crear una copia de la imagen original para no modificar la original
    imagen_original = imagen.copy()
    
    # Transforma el blanco en azul
    rango_blanco_min = np.array([200, 200, 200], dtype="uint8")
    rango_blanco_max = np.array([255, 255, 255], dtype="uint8")
    
    mascara_blanca = cv2.inRange(imagen, rango_blanco_min, rango_blanco_max)
    imagen[mascara_blanca > 0] = [255, 0, 0]  # Azul en BGR
    
    # Convierte la imagen a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplica un umbral para convertir la imagen a binaria
    _, th = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Encuentra contornos en la imagen binaria
    contornos, hierarchy = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    print('hierarchy=', hierarchy)
    
    # Crear una máscara en blanco para rellenar los contornos hijos
    mascara_hijos = np.zeros_like(gray)
    
    # Dibuja todos los contornos en la imagen original
    for i in range(len(contornos)):
        area = cv2.contourArea(contornos[i])
        if area > 3000:
            cv2.drawContours(imagen, contornos, i, (0, 255, 0), 3)
            # Rellena los contornos hijos de blanco
            if hierarchy[0][i][3] != -1:  # Verifica si el contorno tiene un padre
                cv2.drawContours(mascara_hijos, contornos, i, 255, -1)  # Rellena el contorno hijo de blanco
    
    # Usar la máscara para extraer la parte del contorno hijo de la imagen original
    resultado = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_hijos)
    
    # Guarda el resultado en un archivo .png
    cv2.imwrite('resultado.png', resultado)
    
    # Muestra la imagen con el blanco transformado a azul
    cv2.imshow('Imagen con blanco transformado a azul', imagen)
    
    # Muestra la imagen en escala de grises
    cv2.imshow('Imagen en escala de grises', gray)
    
    # Muestra la imagen binaria
    cv2.imshow('Imagen binaria', th)
    
    # Muestra la imagen con contornos
    cv2.imshow('Imagen con contornos', imagen)
    
    # Muestra la máscara de contornos hijos
    cv2.imshow('Máscara de contornos hijos', mascara_hijos)
    
    # Muestra el resultado final con los colores originales del contorno hijo
    cv2.imshow('Resultado', resultado)
    
    # Espera hasta que se presione una tecla
    cv2.waitKey(0)
    cv2.destroyAllWindows()
