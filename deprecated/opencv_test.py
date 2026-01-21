#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

#%%

# Load the image
image = cv2.imread('bucket/img20250313_10443025.png', cv2.IMREAD_GRAYSCALE)

# Create a mask for the lines to be filled (e.g., drawing white lines on a black background)
# This is a simplified example; in a real scenario, you'd generate this mask based on line detection or user input.
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.line(mask, (50, 50), (200, 200), 255, 5) # Example line to be filled

# Perform inpainting using the Telea algorithm
filled_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

plt.imshow(filled_image)
plt.show()

plt.imshow(image)
plt.show()

#%%


# Ler imagem original
img = cv2.imread('bucket/img20250313_10443025.png')

# Converter para tons de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar falhas como buracos (binarização inversa)
_, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Inpaint para preencher falhas
# O valor 3 é o raio de inpainting, pode ajustar
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

plt.imshow(result)
plt.show()

plt.imshow(image)
plt.show()

#%%

# Ler imagem original
img = cv2.imread('bucket/img20250313_10443025.png',cv2.IMREAD_GRAYSCALE)
cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = px.imshow(img)
fig.show()


