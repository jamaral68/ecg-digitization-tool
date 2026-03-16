import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def plot_stages(
    stages: list[tuple[str, np.ndarray]]
    | list[tuple[str, np.ndarray, np.ndarray | None]],
    figsize=(20, 12),
    cols=3,
    point_color="blue",
    point_size=100,
):
    """
    Parameters
    ----------
    stages : list of tuples, each being either:
        - (title, image)
        - (title, image, points)  where points is an Nx2 array of (x, y) or None
    figsize : figure size
    cols : number of columns in the grid
    point_color : color for scatter points
    point_size : marker size for scatter points
    """
    n = len(stages)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for ax, stage in zip(axes, stages):
        title = stage[0]
        img = stage[1]
        points = stage[2] if len(stage) > 2 else None
        draw_contour = stage[3] if len(stage) > 3 else False

        display_img = img.copy()

        if points is not None and len(points) > 0:
            pts = np.asarray(points)
            if draw_contour and title != "corners":
                contour = pts.reshape(-1, 2).astype(np.int32)
                cv2.drawContours(display_img, [contour], -1, (20, 20, 255), 2)

        if display_img.ndim == 3:
            ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(display_img, cmap="gray")

        if points is not None and len(points) > 0:
            pts_2d = pts.reshape(-1, 2)
            ax.scatter(
                pts_2d[:, 0], pts_2d[:, 1], c=point_color, s=point_size, zorder=5
            )
            for idx, (x, y) in enumerate(pts_2d):
                ax.annotate(
                    f"{idx}: ({x}, {y})",
                    (x, y),
                    textcoords="offset points",
                    xytext=(10, -10),
                    fontsize=8,
                    color=point_color,
                    zorder=6,
                )

        ax.set_title(title, fontsize=12)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    plt.show()
    return fig
