import cv2

# Função de callback para capturar a região selecionada
def select_roi(event, x, y, flags, param):
    global x1, y1, x2, y2, cropping, image, roi_selected

    if event == cv2.EVENT_LBUTTONDOWN:  # Quando o botão esquerdo do mouse é pressionado
        x1, y1 = x, y
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:  # Enquanto arrasta o mouse
        temp_image = image.copy()
        cv2.rectangle(temp_image, (x1, y1), (x, y), (0, 255, 0), 2)
        cv2.imshow("Imagem", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:  # Quando o botão esquerdo do mouse é solto
        x2, y2 = x, y
        cropping = False
        roi_selected = True

        # Mostra a região selecionada
        roi = image[y1:y2, x1:x2]
        cv2.imshow("Região Selecionada", roi)
        cv2.imwrite("roi_selecionada.jpg", roi)  # Salva a ROI em um arquivo
        print("Região selecionada salva como 'roi_selecionada.jpg'")

# Inicialize variáveis globais
x1, y1, x2, y2 = 0, 0, 0, 0
cropping = False
roi_selected = False

# Carregue a imagem

image_path = "bucket/ecg_test.png"  # Substitua pelo caminho da imagem
image = cv2.imread(image_path)

if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho.")
else:
    cv2.imshow("Imagem", image)
    cv2.setMouseCallback("Imagem", select_roi)

    print("Use o mouse para selecionar uma área na imagem.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
