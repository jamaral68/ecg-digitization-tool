# USAGE:
# python scan.py (--images <IMG_DIR> | --image <IMG_PATH>) [-i]
# For example, to scan a single image with interactive mode:
# python scan.py --image sample_images/desk.JPG -i
# To scan all images in a directory automatically:
# python scan.py --images sample_images

# Scanned images will be output to directory named 'output'
from pathlib import Path

from scipy.spatial import distance as dist

# import polygon_interacter as poly_i
import numpy as np
import cv2
from matplotlib.lines import Line2D
from matplotlib.artist import Artist
import argparse
import os

from dataclasses import dataclass


@dataclass
class ECGScannerConfig:
    output_dir: Path | None = None
    interactive: bool = False
    min_quad_area_ratio: float = 0.3
    max_quad_angle_range: int = 25
    rescaled_height: float = 500.0
    min_dist: int = 150
    gaussian_blur_kernel_size: int = 5
    morph_kernel_size: int = 11
    canny_sigma: float = 0.2
    rescaled_height: float = 500.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    bilateral_filter_d: int = 5
    bilateral_filter_sigma_color: float = 50
    bilateral_filter_sigma_space: float = 50
    debug_mode: bool = True
    hsv_hue_multiplier: float = 1.0
    hsv_saturation_multiplier: float = 1.5
    hsv_brightness_multiplier: float = 1.2


class ECGScanner(object):
    """An ECG Image Scanner"""

    def __init__(
        self,
        v_margin=60,
        s_margin=90,
        fill_value=255,
        dark_percentile=10,
        s_quantile_offset=0.4,
    ):
        """
        Args:
            interactive (boolean): If True, user can adjust screen contour before
                transformation occurs in interactive pyplot window.
            MIN_QUAD_AREA_RATIO (float): A contour will be rejected if its corners
                do not form a quadrilateral that covers at least MIN_QUAD_AREA_RATIO
                of the original image. Defaults to 0.25.
            MAX_QUAD_ANGLE_RANGE (int):  A contour will also be rejected if the range
                of its interior angles exceeds MAX_QUAD_ANGLE_RANGE. Defaults to 40.
        """

        self.v_margin = v_margin
        self.s_margin = s_margin
        self.fill_value = fill_value
        self.dark_percentile = dark_percentile
        self.s_quantile_offset = s_quantile_offset

    @staticmethod
    def order_points(pts):
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")

    def four_point_transform(self, image, pts):
        # obtain a consistent order of the points and unpack them
        # individually
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left
        # order
        dst = np.array(
            [
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1],
            ],
            dtype="float32",
        )

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # return the warped image
        return warped

    @staticmethod
    def _inpaint_labels(
        image: np.ndarray,
        label_boxes: list[tuple[int, int, int, int]],
        dilate_px: int = 3,
        inpaint_radius: int = 3,
    ) -> np.ndarray:
        """Aplica cv2.inpaint sobre as regiões de texto detectadas pelo YOLO de labels."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for lx1, ly1, lx2, ly2 in label_boxes:
            x1 = max(0, lx1 - dilate_px)
            y1 = max(0, ly1 - dilate_px)
            x2 = min(w, lx2 + dilate_px)
            y2 = min(h, ly2 + dilate_px)
            mask[y1:y2, x1:x2] = 255
        return cv2.inpaint(image, mask, inpaint_radius, cv2.INPAINT_TELEA)



    @staticmethod
    def detect_signal_profile(img_bgr, dark_percentile=10, s_quantile_offset=0.4):
        """Detecta o perfil HSV do traçado do ECG.

        O sinal é tipicamente o conjunto de pixels mais escuros (tinta sobre
        papel) e acromático (preto/cinza/azul-escuro). Estratégia:
        1. Pega os pixels mais escuros (abaixo do percentil `dark_percentile` de V).
        2. Calcula estatísticas de S e V desse conjunto → tolerâncias para a máscara.
        """
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v = hsv[..., 2]
        v_thresh = float(np.percentile(v, dark_percentile))
        dark = hsv[v <= v_thresh]
        if dark.size == 0:
            return None
        return {
            "v_thresh": v_thresh,
            "s_median": float(np.median(dark[:, 1])),
            "v_median": float(np.median(dark[:, 2])),
            "s_max": float(np.quantile(dark[:, 1], 0.5 + s_quantile_offset)),
        }

    def keep_signal_color(
        self,
        img_bgr,
    ):
        """Mantém apenas pixels com perfil do traçado; o resto vira branco.

        Define a cor do sinal como (V baixo, S baixo) e aceita pixels com
        V <= v_thresh + v_margin e S <= s_max + s_margin. Tudo fora dessa
        faixa (grade colorida, fundo branco, texto colorido) é substituído.
        """
        if img_bgr.ndim == 2:
            return img_bgr.copy(), None, None

        profile = self.detect_signal_profile(
            img_bgr,
            dark_percentile=self.dark_percentile,
            s_quantile_offset=self.s_quantile_offset,
        )
        if profile is None:
            return img_bgr.copy(), None, None

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        s_upper = min(255, int(profile["s_max"] + self.s_margin))
        v_upper = min(255, int(profile["v_thresh"] + self.v_margin))
        signal_mask = cv2.inRange(hsv, (0, 0, 0), (180, s_upper, v_upper))

        cleaned = np.full_like(img_bgr, self.fill_value)
        cleaned[signal_mask > 0] = img_bgr[signal_mask > 0]
        return cleaned



    @staticmethod
    def _block_median_binarize(
        gray: np.ndarray,
        block_size: int = 50,
        threshold: int = 25,
        pre_median_k: int = 3,
        pre_gaussian_k: int = 3,
        post_median_k: int = 5,
    ) -> np.ndarray:
        """Binariza um crop via threshold adaptativo por blocos sobre a mediana local.

        Parâmetros calibráveis:
          block_size    — tamanho do bloco em pixels; blocos menores adaptam melhor
                          a variações locais de iluminação, mas são mais lentos.
          threshold     — diferença máxima entre pixel e mediana do bloco para ser
                          considerado sinal; reduzir = mais seletivo (menos ruído),
                          aumentar = mais permissivo (recupera traços tênues).
          pre_median_k  — kernel do medianBlur antes de inverter; remove salt-and-
                          pepper sem borrar bordas. Deve ser ímpar >= 1.
          pre_gaussian_k— kernel do GaussianBlur antes de inverter; suaviza
                          gradientes de iluminação. Deve ser ímpar >= 1.
          post_median_k — kernel do medianBlur na saída; elimina pontos isolados
                          no binarizado. Deve ser ímpar >= 1.
        """
        def _get_block_index(shape, yx, bs):
            y = np.arange(max(0, yx[0] - bs), min(shape[0], yx[0] + bs))
            x = np.arange(max(0, yx[1] - bs), min(shape[1], yx[1] + bs))
            return tuple(np.meshgrid(y, x))

        def _adaptive_median_threshold(block):
            med = np.median(block)
            out = np.zeros_like(block)
            out[block - med < threshold] = 255
            return out

        img = gray.copy()
        if pre_median_k > 1:
            img = cv2.medianBlur(img, pre_median_k | 1)
        if pre_gaussian_k > 1:
            k = pre_gaussian_k | 1
            img = cv2.GaussianBlur(img, (k, k), 0)
        img = 255 - img

        out = np.zeros_like(img)
        for row in range(0, img.shape[0], block_size):
            for col in range(0, img.shape[1], block_size):
                idx = _get_block_index(img.shape, (row, col), block_size)
                out[idx] = _adaptive_median_threshold(img[idx])

        if post_median_k > 1:
            out = cv2.medianBlur(out, post_median_k | 1)
        return out


    def scan_yolo(
        self,
        image: np.ndarray,
        lead_boxes: dict[str, tuple[int, int, int, int]],
        label_boxes: list[tuple[int, int, int, int]] | None = None,
        binarize_method: str = "adaptive",
        block_size: int = 50,
        block_threshold: int = 25,
        pre_median_k: int = 3,
        pre_gaussian_k: int = 3,
        post_median_k: int = 5,
    ) -> dict[str, np.ndarray]:
        """
        binarize_method:
          "adaptive"     — adaptiveThreshold Gaussian (padrão)
          "block_median" — threshold adaptativo por blocos sobre a mediana local
          "morph_lines"  — remoção de grade via morfologia antes de binarizar
        """
        orig = self.keep_signal_color(image.copy())
        if label_boxes:
            orig = self._inpaint_labels(orig, label_boxes)
        image_boxes: dict[str, np.ndarray] = dict()
        for lead_name, screenCnt in lead_boxes.items():
            if lead_name in image_boxes:
                continue
            warped = self.four_point_transform(orig, screenCnt)
            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

            if binarize_method == "block_median":
                image_boxes[lead_name] = self._block_median_binarize(
                    gray,
                    block_size=block_size,
                    threshold=block_threshold,
                    pre_median_k=pre_median_k,
                    pre_gaussian_k=pre_gaussian_k,
                    post_median_k=post_median_k,
                )
            elif binarize_method == "morph_lines":
                image_boxes[lead_name] = self._morph_lines_binarize(gray)
            else:
                sharpen = cv2.GaussianBlur(gray, (0, 0), 3)
                sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)
                image_boxes[lead_name] = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15
                )
        return image_boxes


class PolygonInteractor(object):
    """
    An polygon editor
    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError(
                "You must first add the polygon to a figure or canvas before defining the interactor"
            )
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy)
        self.line = Line2D(x, y, marker="o", markerfacecolor="r", animated=True)
        self.ax.add_line(self.line)

        # cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.draw_callback)
        canvas.mpl_connect("button_press_event", self.button_press_callback)
        canvas.mpl_connect("button_release_event", self.button_release_callback)
        canvas.mpl_connect("motion_notify_event", self.motion_notify_callback)
        self.canvas = canvas

    def get_poly_points(self):
        return np.asarray(self.poly.xy)

    def draw_callback(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        "this method is called whenever the polygon object is called"
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        "get the index of the vertex under point if within epsilon tolerance"

        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def button_press_callback(self, event):
        "whenever a mouse button is pressed"
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def button_release_callback(self, event):
        "whenever a mouse button is released"
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def motion_notify_callback(self, event):
        "on mouse movement"
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--images", help="Directory of images to be scanned")
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument(
        "-i",
        action="store_true",
        help="Flag for manually verifying and/or setting document corners",
    )

    args = vars(ap.parse_args())
    im_dir = args["images"]
    im_file_path = args["image"]
    interactive_mode = args["i"]

    scanner = ECGScanner(interactive=interactive_mode)

    valid_formats = [".jpg", ".jpeg", ".jp2", ".png", ".bmp", ".tiff", ".tif"]

    def get_ext(f):
        return os.path.splitext(f)[1].lower()

    # Scan single image specified by command line argument --image <IMAGE_PATH>
    if im_file_path:
        scanner.scan(im_file_path)

    # Scan all valid images in directory specified by command line argument --images <IMAGE_DIR>
    else:
        im_files = [f for f in os.listdir(im_dir) if get_ext(f) in valid_formats]
        for im in im_files:
            scanner.scan(im_dir + "/" + im)
