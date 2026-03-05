
import numpy as np

from PIL import Image

# Caminho para a imagem
image_path = "tiny_10433218.png"  # Substitua pelo caminho do arquivo de imagem

# Carregar a imagem
image = Image.open(image_path)

# Solicitar o ângulo de rotação
angle = float(input("Digite o ângulo de rotação (em graus): "))

# Rotacionar a imagem
rotated_image = image.rotate(angle, expand=True)

# Salvar a imagem rotacionada em um novo arquivo
rotated_image.save("imagem_rotacionada.png")

print("Imagem rotacionada salva como 'imagem_rotacionada.jpg'.")
