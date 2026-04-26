import io
import zipfile

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.signal import medfilt


def extract_curve_robust(
    img_lead: np.ndarray,
    dilate_iters: int = 1,
    medfilt_kernel: int = 11,
    mad_factor: float = 5.0,
) -> np.ndarray:
    """Extrai a curva (yseg) de um crop binário descartando ruído.

    Pipeline:
      1. Filtra o ruído estrutural mantendo apenas o **maior componente
         conexo** escuro (descarta resíduo de grade, manchas, textos pequenos).
         Aplica dilatação leve antes para reconectar fragmentos do traço.
      2. Faz `argmin` coluna-a-coluna na imagem filtrada.
      3. Remove outliers pontuais comparando cada `yseg[x]` com a mediana
         móvel; substitui valores com desvio > `mad_factor * MAD` por
         interpolação linear dos vizinhos válidos.
    """
    binary = img_lead if img_lead.ndim == 2 else cv.cvtColor(img_lead, cv.COLOR_BGR2GRAY)
    inv = cv.bitwise_not(binary)
    if dilate_iters > 0:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        inv = cv.dilate(inv, kernel, iterations=dilate_iters)

    n_labels, labels, stats, _ = cv.connectedComponentsWithStats(inv, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        signal_only = np.where(labels == largest, 0, 255).astype(np.uint8)
    else:
        signal_only = binary

    yseg = np.argmin(signal_only, axis=0)

    kernel_size = max(3, medfilt_kernel | 1)  # garante ímpar e >= 3
    yseg_smooth = medfilt(yseg, kernel_size=kernel_size)
    residual = np.abs(yseg.astype(float) - yseg_smooth.astype(float))
    mad = np.median(residual)
    if mad > 0:
        outliers = residual > mad_factor * mad
        if outliers.any() and (~outliers).any():
            valid_idx = np.flatnonzero(~outliers)
            outlier_idx = np.flatnonzero(outliers)
            yseg = yseg.copy()
            yseg[outlier_idx] = np.interp(outlier_idx, valid_idx, yseg[valid_idx])

    return yseg.astype(int)


def remove_labels_inpaint(
    img_gray: np.ndarray, label_boxes: list, x1_lead: int, y1_lead: int
) -> np.ndarray:
    """
    Remove text labels from a grayscale ECG crop using inpainting.

    For each label bounding box that overlaps the lead crop region, a mask is
    created and cv2.inpaint reconstructs the underlying signal pixels.
    """
    crop = img_gray[y1_lead:, x1_lead:].copy()
    h, w = crop.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for lx1, ly1, lx2, ly2 in label_boxes:
        cx1 = max(lx1 - x1_lead, 0)
        cy1 = max(ly1 - y1_lead, 0)
        cx2 = min(lx2 - x1_lead, w)
        cy2 = min(ly2 - y1_lead, h)
        if cx1 >= cx2 or cy1 >= cy2:
            continue
        mask[cy1:cy2, cx1:cx2] = 255

    if mask.any():
        crop = cv.inpaint(crop, mask, inpaintRadius=7, flags=cv.INPAINT_TELEA)

    return crop


def convert_to_secmv(xs, ys, wp, pulse_per_sec, pulse_per_mv):
    """
    Convert ECG waveform pixel coordinates to physical units (seconds, mV).
    """
    ys_smooth = medfilt(ys, kernel_size=min(len(ys) // 2 * 2 + 1, 101))
    baseline_px = np.percentile(ys_smooth, 50)
    ymv = (baseline_px - ys) / pulse_per_mv
    xsec = np.array(xs) * (pulse_per_sec / wp)
    return xsec, ymv


def interpolate_segment(x, y, num):
    """
    Interpolate an ECG segment to a fixed number of points using cubic spline.
    """
    x_interp = np.linspace(0, 1, len(x))
    f = interpolate.CubicSpline(x_interp, y)
    x_new = np.linspace(0, 1, num)
    y_new = f(x_new)
    return x_new, y_new


def segment_to_df(ecg_curves, pulse_per_sec, pulse_per_mv, num_pts):
    """
    Convert a list of ECG waveform segments into a pandas DataFrame.
    Each column represents one lead's interpolated ECG signal.
    """
    dfs = []
    for lead_name, curve in ecg_curves.items():
        xsec, ymv = convert_to_secmv(
            curve["xseg"], curve["yseg"], curve["wpulse"], pulse_per_sec, pulse_per_mv
        )
        x_new, y_new = interpolate_segment(xsec, ymv, num_pts)
        print(len(xsec), len(y_new))
        dfs.append(pd.Series(y_new, name=lead_name, index=x_new))
    return pd.concat(dfs,axis=1)


def draw_overlay(image_path, result, model, label_boxes=None):
    """
    Draw ECG waveforms on top of the original image for visual validation.

    Replica o pipeline de `ecg_to_csv`: encolhe o crop em 10 px em X, aplica
    inpainting dos labels detectados (se houver) e extrai a curva via argmin
    na grayscale tratada — assim a sobreposição reflete o sinal realmente
    digitalizado, não uma versão crua do bbox.
    """
    img = cv.imread(image_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    overlay = img.copy()

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        lead_name = model.names[cls_id]

        if lead_name.lower() == "pulse":
            continue

        crop_x1 = max(0, x1 + 10)
        crop_x2 = min(img_gray.shape[1], x2 - 10)

        if label_boxes:
            full_crop = remove_labels_inpaint(
                img_gray, label_boxes, x1_lead=crop_x1, y1_lead=y1
            )
            crop = full_crop[: (y2 - y1), : (crop_x2 - crop_x1)]
        else:
            crop = img_gray[y1:y2, crop_x1:crop_x2]

        if crop.shape[0] < 2 or crop.shape[1] < 2:
            continue

        yseg = np.argmin(crop, axis=0)
        xseg = np.arange(len(yseg))

        color_map = {"I": (255, 0, 0), "II": (0, 255, 0), "III": (0, 0, 255)}
        color = color_map.get(lead_name, (0, 0, 255))

        cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for k in range(len(xseg) - 1):
            pt1 = (crop_x1 + int(xseg[k]), y1 + int(yseg[k]))
            pt2 = (crop_x1 + int(xseg[k + 1]), y1 + int(yseg[k + 1]))
            cv.line(overlay, pt1, pt2, color, 2)

    alpha = 0.7
    final = cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return final


def draw_overlay_from_curves(curves_df, line_color=(0, 0, 255), thickness=2):
    """
    Desenha cada curva digitalizada sobre o seu próprio recorte (imagem do lead
    já endireitada pelo `ECGScanner`).

    Como `ecg_to_csv` agora extrai `(xseg, yseg)` no espaço do crop warped — e
    não da imagem original — não dá para desenhar diretamente sobre a foto
    crua sem aplicar a inversa da perspectiva. Esta função plota a curva em
    cima de cada recorte e devolve um dict `{lead_name: overlay_bgr}`.

    Espera-se que `curves_df` tenha uma linha por lead, com as colunas:
      - name (str): nome do lead.
      - xseg, yseg (array-like): curva em pixels relativos ao crop.
      - rec (np.ndarray): imagem do crop (saída de `scan_yolo`).
    """
    overlays = {}
    for _, row in curves_df.iterrows():
        rec = row["rec"]
        if rec is None:
            continue
        canvas = rec if rec.ndim == 3 else cv.cvtColor(rec, cv.COLOR_GRAY2BGR)
        canvas = canvas.copy()

        xseg = np.asarray(row["xseg"])
        yseg = np.asarray(row["yseg"])
        if len(xseg) < 2 or len(yseg) < 2:
            overlays[row["name"]] = canvas
            continue

        for k in range(len(xseg) - 1):
            pt1 = (int(xseg[k]), int(yseg[k]))
            pt2 = (int(xseg[k + 1]), int(yseg[k + 1]))
            cv.line(canvas, pt1, pt2, line_color, thickness)

        overlays[row["name"]] = canvas
    return overlays


def line_list_to_curves_df(line_list):
    """Converte o `line_list` de `ecg_to_csv` em DataFrame para `draw_overlay_from_curves`.

    Lê os campos efetivamente populados pelo `ecg_to_csv` atual:
    `name`, `xseg`, `yseg` e `rec` (recorte warped do lead).
    """
    rows = []
    for line in line_list:
        for seg in line["curves"]:
            rows.append(
                {
                    "name": seg["name"],
                    "xseg": seg["xseg"],
                    "yseg": seg["yseg"],
                    "rec": seg.get("rec"),
                }
            )
    return pd.DataFrame(rows)


def plot_ecg_signal(time, signal, ax):
    """
    Plot a single ECG signal on a Matplotlib axis with grid and axis labels.
    """
    ax.plot(time, signal)
    ax.set_xticks(np.arange(int(time[0]), round(time[-1]) + 1))
    ax.minorticks_on()
    ax.grid(which="major", linestyle="-", color="red", linewidth=1.0)
    ax.grid(which="minor", linestyle=":", color="black", linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    return ax


def plot_ecg(df, columns, title, n_rows=4, n_columns=4, fs=500, figure_size=(20, 12)):
    """
    Plot multiple ECG leads from a DataFrame in a grid layout.
    """
    fig, axes = plt.subplots(n_rows, n_columns, figsize=figure_size)
    fig.suptitle(title, fontsize=20)

    for idx, col in enumerate(columns):
        row_idx = idx // n_columns
        col_idx = idx % n_columns
        ax = axes[row_idx][col_idx] if n_rows > 1 and n_columns > 1 else axes[idx]
        ts = np.arange(df[col].size) / fs
        plot_ecg_signal(ts, df[col], ax)

    plt.subplots_adjust(top=0.92, hspace=0.45, wspace=0.5)
    return fig


def create_zip(csv_bytes, overlay_img, yolo_img, csv_name):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(csv_name, csv_bytes)
        zip_file.writestr("overlay.png", overlay_img)
        zip_file.writestr("yolo_bbox.png", yolo_img)

    zip_buffer.seek(0)
    return zip_buffer
