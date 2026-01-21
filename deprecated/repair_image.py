#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import scipy
import cv2 as cv
import numpy as np
import pandas as pd
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import skimage as ski
import pytesseract
from edt_utils import is_nan, py_blockproc, display_segments, detect_ref_pulse, print_line_dict,segment_to_df, remove_text
from ss import pattern_match
from edt_utils import process_line,get_values_from_img,measure_extract_pulse ,plot_ecg, extract_image
from scipy.signal import find_peaks
import operator

# Ler a imagem
img = cv.imread('bucket/img20250313_10443025.png', cv.IMREAD_GRAYSCALE)

# Binarizar (ajuste o limiar conforme necessário)
_, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

# Aplicar dilatação para engrossar linhas
kernel = np.ones((3,3), np.uint8)
dilated = cv.dilate(binary, kernel, iterations=1)

# Fechamento para preencher buracos na linha
closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

# Inverter de volta (se quiser)
result = cv.bitwise_not(closed)

plt.imshow(result)
plt.show()
# Salvar
#cv.imwrite('linha_reforçada.png', result)

