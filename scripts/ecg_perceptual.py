"""
ECG Restoration: U-Net + Masked MAE + SSIM + Edge + Perceptual (VGG19)
- Aceita imagens grayscale de qualquer tamanho (batch_size=1)
- Entrada: [imagem_noisy, mask_valid] -> Saída: imagem_restaurada
- Avaliação: PSNR e SSIM
"""
import os, glob, random, math
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# =========================================================
# 1) Geração de ruído e falhas estruturais + máscara
# =========================================================
def add_salt_pepper(img, amount=0.004):
    noisy = img.copy()
    n = img.size
    num_salt = int(amount * n * 0.5)
    num_pepper = int(amount * n * 0.5)
    # sal
    ys = np.random.randint(0, img.shape[0], num_salt)
    xs = np.random.randint(0, img.shape[1], num_salt)
    noisy[ys, xs] = 255
    # pimenta
    ys = np.random.randint(0, img.shape[0], num_pepper)
    xs = np.random.randint(0, img.shape[1], num_pepper)
    noisy[ys, xs] = 0
    return noisy

def add_gaussian_noise(img, sigma=12):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)

def add_random_gridlines(img, p=0.3):
    """Simula grade do papel ou artefatos de varredura"""
    if random.random() > p:
        return img
    h, w = img.shape
    out = img.copy()
    # horizontais
    step_h = random.choice([16, 20, 25, 32])
    for y in range(0, h, step_h):
        cv2.line(out, (0, y), (w-1, y), color=(random.randint(170, 220),), thickness=1)
    # verticais finas
    step_w = random.choice([16, 20, 25, 32])
    for x in range(0, w, step_w):
        cv2.line(out, (x, 0), (x, h-1), color=(random.randint(170, 220),), thickness=1)
    return out

def add_thick_scratches(img, count=2):
    """Riscos grossos pretos/cinzas (cabo flat do scanner, dedos, etc.)"""
    out = img.copy()
    h, w = img.shape
    for _ in range(random.randint(0, count)):
        x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
        x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
        thickness = random.randint(2, 6)
        color = random.randint(0, 60)
        cv2.line(out, (x1, y1), (x2, y2), color=(color,), thickness=thickness)
    return out

def random_erasure(img, max_boxes=3):
    """Cortes (apagões brancos) + retorna máscara (1=válido, 0=faltante)"""
    out = img.copy()
    mask = np.ones_like(img, dtype=np.uint8)
    h, w = img.shape
    for _ in range(random.randint(0, max_boxes)):
        bw = random.randint(max(6, w//20), max(8, w//8))
        bh = random.randint(max(6, h//20), max(8, h//8))
        x1 = random.randint(0, max(1, w - bw))
        y1 = random.randint(0, max(1, h - bh))
        x2, y2 = x1 + bw, y1 + bh
        cv2.rectangle(out, (x1, y1), (x2, y2), color=(255,), thickness=-1)  # apagão claro
        mask[y1:y2, x1:x2] = 0
    return out, mask

def subtle_warp(img, p=0.4):
    """Leve warping/deskew para simular deformação de papel"""
    if random.random() > p:
        return img
    h, w = img.shape
    # pequena rotação e shear
    angle = random.uniform(-2.0, 2.0)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out

def make_noisy_and_mask(img):
    x = img
    x = add_gaussian_noise(x, sigma=random.choice([8, 12, 16]))
    x = add_salt_pepper(x, amount=random.choice([0.002, 0.0035, 0.005]))
    x = add_random_gridlines(x, p=0.35)
    x = add_thick_scratches(x, count=3)
    x = subtle_warp(x, p=0.5)
    x, mask_missing = random_erasure(x, max_boxes=3)
    # máscara de validade (1 = válido, 0 = faltante); também penalizaremos riscos grossos com MAE normal
    valid_mask = mask_missing
    return x, valid_mask

# =========================================================
# 2) Loader -> tf.data com batch_size=1 (tamanho livre)
# =========================================================
def load_clean_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")))
    imgs = []
    for p in paths:
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None: 
            continue
        # opcional: recorte de bordas brancas muito grandes
        imgs.append(g)
    return imgs

def normalize01(x):
    return (x.astype(np.float32) / 255.0)

def data_gen(clean_imgs):
    while True:
        img = random.choice(clean_imgs)
        noisy, mask = make_noisy_and_mask(img)
        yield (normalize01(noisy)[..., None], normalize01(mask)[..., None]), normalize01(img)[..., None]

def make_dataset(clean_imgs, steps_per_epoch=200):
    ds = tf.data.Dataset.from_generator(
        lambda: data_gen(clean_imgs),
        output_signature=(
            (tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
             tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32)
        )
    )
    # batch=1 para tamanhos variáveis
    ds = ds.map(lambda x, y: ((tf.expand_dims(x[0], 0), tf.expand_dims(x[1], 0)), tf.expand_dims(y, 0)))
    ds = ds.take(steps_per_epoch).prefetch(2)
    return ds

# =========================================================
# 3) Modelo: U-Net (entrada imagem + máscara)
# =========================================================
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(f, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def unet_flexible():
    noisy_in = layers.Input(shape=(None, None, 1), name="noisy")
    mask_in  = layers.Input(shape=(None, None, 1), name="mask")
    x = layers.Concatenate()([noisy_in, mask_in])  # [B,H,W,2]

    c1 = conv_block(x, 32); p1 = layers.MaxPooling2D(2)(c1)
    c2 = conv_block(p1, 64); p2 = layers.MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128); p3 = layers.MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 256)

    u3 = layers.UpSampling2D(2)(c4); u3 = layers.Concatenate()([u3, c3]); c5 = conv_block(u3, 128)
    u2 = layers.UpSampling2D(2)(c5); u2 = layers.Concatenate()([u2, c2]); c6 = conv_block(u2, 64)
    u1 = layers.UpSampling2D(2)(c6); u1 = layers.Concatenate()([u1, c1]); c7 = conv_block(u1, 32)

    out = layers.Conv2D(1, 3, padding="same", activation="sigmoid", name="restored")(c7)
    return models.Model([noisy_in, mask_in], out, name="UNet_ECG")

# =========================================================
# 4) Loss: Masked MAE + SSIM + Edge + Perceptual (VGG19)
# =========================================================
# — Sobel edge loss
def sobel_edges(x):
    # tf.image.sobel_edges -> [B,H,W,1,2] (dx, dy)
    e = tf.image.sobel_edges(x)
    dx, dy = e[..., 0], e[..., 1]
    mag = tf.sqrt(tf.square(dx) + tf.square(dy) + 1e-6)
    return mag

# — Perceptual: usa VGG19 (grayscale replicada para 3 canais), bloqueio 3
VGG = None
def build_vgg():
    global VGG
    base = tf.keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(None, None, 3))
    # Congela
    base.trainable = False
    # Saída em um nível intermediário
    layer_name = "block3_conv3"
    vgg = models.Model(base.input, base.get_layer(layer_name).output)
    VGG = vgg

@tf.function
def perceptual_features(x01):
    # espera [B,H,W,1] em 0..1
    x3 = tf.concat([x01, x01, x01], axis=-1)  # grayscale -> RGB
    x3 = tf.keras.applications.vgg19.preprocess_input(x3*255.0)  # VGG19 expects 0..255 BGR-centered
    return VGG(x3)

def ssim_loss(y_true, y_pred):
    # tf.image.ssim retorna maior = melhor; loss = 1 - ssim normalizado em [0,1]
    s = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - tf.reduce_mean((s + 1.0) / 2.0)

def masked_mae(y_true, y_pred, mask_valid):
    # somente onde mask==1 (válido) + peso menor nas regiões faltantes para guiar suavemente
    eps = 1e-6
    w_valid = mask_valid
    w_missing = 1.0 - mask_valid
    # pesos: válido = 1.0, faltante = 0.3 (ajuda a não “lavar” o sinal)
    w = w_valid + 0.3 * w_missing
    num = tf.reduce_sum(w * tf.abs(y_true - y_pred))
    den = tf.reduce_sum(w) + eps
    return num / den

@tf.function
def composite_loss(y_true, y_pred, mask_valid):
    # 1) masked MAE
    l_mae = masked_mae(y_true, y_pred, mask_valid)
    # 2) SSIM loss
    l_ssim = ssim_loss(y_true, y_pred)
    # 3) Edge loss (MAE nos mapas de borda)
    e_true = sobel_edges(y_true)
    e_pred = sobel_edges(y_pred)
    l_edge = tf.reduce_mean(tf.abs(e_true - e_pred))
    # 4) Perceptual loss
    f_true = perceptual_features(y_true)
    f_pred = perceptual_features(y_pred)
    l_perc = tf.reduce_mean(tf.abs(f_true - f_pred))
    # Pesos (ajuste fino depois do primeiro treino)
    return 0.55*l_mae + 0.2*l_ssim + 0.15*l_edge + 0.10*l_perc, (l_mae, l_ssim, l_edge, l_perc)

# =========================================================
# 5) Treino
# =========================================================
def train(
    clean_folder="ecg_clean",
    epochs=10,
    steps_per_epoch=300,
    val_steps=60,
    mixed_precision=True
):
    if mixed_precision:
        try:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception:
            pass

    clean_imgs = load_clean_images(clean_folder)
    assert len(clean_imgs) > 0, "Nenhuma imagem encontrada em ecg_clean/*.png|jpg"

    build_vgg()
    model = unet_flexible()

    # Otimizador
    opt = tf.keras.optimizers.Adam(1e-4)

    # Datasets
    train_ds = make_dataset(clean_imgs, steps_per_epoch=steps_per_epoch)
    val_ds   = make_dataset(clean_imgs, steps_per_epoch=val_steps)

    # Checkpoints
    ckpt = tf.train.Checkpoint(model=model, optimizer=opt)
    manager = tf.train.CheckpointManager(ckpt, "./ckpts_ecg", max_to_keep=3)

    # Treino manual para logar perdas parciais
    train_loss = tf.keras.metrics.Mean()
    val_loss   = tf.keras.metrics.Mean()

    @tf.function
    def train_step(noisy, mask, gt):
        with tf.GradientTape() as tape:
            pred = model([noisy, mask], training=True)
            loss, parts = composite_loss(gt, pred, mask)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, parts, pred

    @tf.function
    def val_step(noisy, mask, gt):
        pred = model([noisy, mask], training=False)
        loss, parts = composite_loss(gt, pred, mask)
        return loss, parts, pred

    best = 1e9
    for ep in range(1, epochs+1):
        train_loss.reset_states()
        val_loss.reset_states()

        # -------- train
        for (x, m), y in train_ds:
            loss, parts, _ = train_step(x, m, y)
            train_loss.update_state(loss)

        # -------- val
        for (x, m), y in val_ds:
            vloss, vparts, _ = val_step(x, m, y)
            val_loss.update_state(vloss)

        print(f"Epoch {ep:02d} | train {train_loss.result():.4f} | val {val_loss.result():.4f}")
        # checkpoint
        if float(val_loss.result()) < best:
            best = float(val_loss.result())
            manager.save()
            model.save("./ecg_unet_perceptual.keras")
            print("  ↳ modelo salvo (melhor até agora)")

    return model

# =========================================================
# 6) Avaliação objetiva (PSNR/SSIM) e restauração
# =========================================================
def evaluate_and_restore(model_path="./ecg_unet_perceptual.keras",
                         test_folder="ecg_clean",
                         max_samples=6,
                         save_examples=True):
    # Carrega modelo
    custom_objects = {}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    build_vgg()  # para perceptual em avaliação se precisar

    paths = sorted(glob.glob(os.path.join(test_folder, "*.png")) + glob.glob(os.path.join(test_folder, "*.jpg")))
    psnrs, ssims = [], []
    os.makedirs("restored_out", exist_ok=True)

    for i, p in enumerate(paths[:max_samples]):
        gt = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        noisy, mask = make_noisy_and_mask(gt)

        x = (noisy.astype(np.float32)/255.0)[None, ..., None]
        m = (mask.astype(np.float32)/255.0)[None, ..., None]
        y_pred = model.predict([x, m], verbose=0)[0, ..., 0]
        y_img = (y_pred*255.0).clip(0,255).astype(np.uint8)

        psnrs.append(psnr(gt, y_img, data_range=255))
        ssims.append(ssim(gt, y_img, data_range=255))

        if save_examples:
            canvas = np.concatenate([noisy, y_img, gt], axis=1)
            cv2.imwrite(f"restored_out/sample_{i:02d}.png", canvas)

    print(f"PSNR médio: {np.mean(psnrs):.2f} dB | SSIM médio: {np.mean(ssims):.4f}")
    print("Exemplos salvos em restored_out/ (noisy | restored | clean)")

# =========================================================
# 7) Uso em uma imagem ruidosa real (sem GT)
# =========================================================
def restore_single_image(model_path, noisy_path, out_path="ecg_restaurada.png"):
    model = tf.keras.models.load_model(model_path, compile=False)
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    # Se não tiver máscara, usamos tudo válido (=1). Para fotos com cortes, você pode
    # gerar uma máscara heurística (ex.: regiões totalmente brancas -> 0).
    mask = np.ones_like(noisy, dtype=np.uint8)
    x = (noisy.astype(np.float32)/255.0)[None, ..., None]
    m = (mask.astype(np.float32)/255.0)[None, ..., None]
    pred = model.predict([x, m], verbose=0)[0, ..., 0]
    out = (pred*255.0).clip(0,255).astype(np.uint8)
    cv2.imwrite(out_path, out)
    print(f"Salvo em {out_path}")

# =========================================================
# 8) Execução (exemplo)
# =========================================================
if __name__ == "__main__":
    # 1) treinar (ajuste epochs conforme dataset)
    model = train(clean_folder="ecg_clean", epochs=12, steps_per_epoch=300, val_steps=60, mixed_precision=True)

    # 2) avaliar rapidamente em amostras sintéticas e salvar exemplos
    evaluate_and_restore(model_path="./ecg_unet_perceptual.keras",
                         test_folder="ecg_clean",
                         max_samples=6,
                         save_examples=True)

    # 3) restaurar uma imagem real (substitua o caminho)
    # restore_single_image("./ecg_unet_perceptual.keras", "ecg_test_noisy.png", "ecg_restaurada.png")
