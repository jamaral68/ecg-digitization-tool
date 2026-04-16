import cv2 as cv
import matplotlib.pyplot as plt
from edt_utils import predict_and_draw

def ecg_to_csv(setup, model, device, label_model=None, save_overlay=True):

    df = None

    img = cv.imread(setup.image)

    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {setup.image}")

    plt.figure()
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title("Imagem original")
    plt.axis("off")
    plt.show()

    # detecção + visualização
    img_out = predict_and_draw(model, img, device, threshold=0.5)


    plt.figure()
    plt.imshow(cv.cvtColor(img_out, cv.COLOR_BGR2RGB))
    plt.title("Detecção ECG")
    plt.axis("off")
    plt.show()



    return df