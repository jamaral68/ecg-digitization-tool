import cv2
import os
import numpy as np

# Folder
input_dir = "input_png"
output_dir = "output_yolo"

# Create folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Parâmetros configuráveis
limiar = 150  # ajuste conforme necessário
kernel_size = (3, 3)  # tamanho do kernel para dilatação
dilation = 5

# Kernel para operação morfológica
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(".png"):
        input_path = os.path.join(input_dir, filename)

        # 1. Ler imagem
        img = cv2.imread(input_path)

        # 2. Converter para escala de cinza
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Binarização com limiar configurável
        _, binary = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)

        inverted = cv2.bitwise_not(binary)

        # 4. Dilatação usando morphologyEx
        dilated = cv2.morphologyEx(inverted,cv2.MORPH_DILATE,np.ones((3,3)),iterations=dilation)
        #cv2.morphologyEx(binary,cv2.MORPH_DILATE,np.ones((3,3)),iterations=dilation)

        # 5. Inverter imagem (fundo preto, sinais brancos)
        

        # 6. Salvar com sufixo _yolo
        name, ext = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_yolo.png")
        cv2.imwrite(output_path, inverted)

        print(f"Convertido: {filename} → {name}_yolo.png")

print("Processamento concluído.")