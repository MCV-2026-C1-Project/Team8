"""Generate illustrative plots of the 1D concatenated descriptors used for retrieval.

Outputs saved to plots/:
  - example_descriptor_cielab.png : Image + 1D concatenated CIELab (8-64-64) descriptor.
  - example_descriptor_hsv.png    : Image + 1D concatenated HSV (16-16-8) descriptor.
"""
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "plots"
DATA_IMG = ROOT / "data" / "qsd1_w1" / "00000.jpg"


def get_sample_image():
    """Finds an existing sample image from QSD1 or BBDD."""
    if DATA_IMG.exists():
        return DATA_IMG
    
    # Fallback to any JPG found in data
    found = list(ROOT.glob("data/**/*.jpg"))
    if not found:
        raise FileNotFoundError("No sample .jpg found in data/ to generate descriptor examples.")
    return found[0]


# -----------------------------------------------------------------------------
# 1. CIELab Descriptor (8, 64, 64)
# -----------------------------------------------------------------------------
def plot_cielab_example(img_bgr, img_name, out_dir):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    bins = (8, 64, 64)

    h_L = cv2.calcHist([lab], [0], None, [bins[0]], [0, 256]).flatten()
    h_a = cv2.calcHist([lab], [1], None, [bins[1]], [0, 256]).flatten()
    h_b = cv2.calcHist([lab], [2], None, [bins[2]], [0, 256]).flatten()

    desc = np.concatenate([h_L, h_a, h_b]).astype(np.float64)
    desc /= desc.sum()  # L1 normalize

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.2, 0.05])

    # Left: Original Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Query Image\n({img_name})", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Right: 1D Concatenated Descriptor
    ax_hist = fig.add_subplot(gs[0, 1])
    x = np.arange(len(desc))

    # Segment slices
    i_L = bins[0]
    i_a = i_L + bins[1]

    ax_hist.bar(x[:i_L], desc[:i_L], color="#4D4D4D", label=f"L: Lightness ({bins[0]} bins)", width=0.85, edgecolor="black", linewidth=0.3)
    ax_hist.bar(x[i_L:i_a], desc[i_L:i_a], color="#D95F02", label=f"a*: Green-Red ({bins[1]} bins)", width=0.85, edgecolor="black", linewidth=0.3)
    ax_hist.bar(x[i_a:], desc[i_a:], color="#1B9E77", label=f"b*: Blue-Yellow ({bins[2]} bins)", width=0.85, edgecolor="black", linewidth=0.3)

    # Dividing lines
    ax_hist.axvline(i_L - 0.5, color="#B22222", linestyle="--", linewidth=1.2)
    ax_hist.axvline(i_a - 0.5, color="#B22222", linestyle="--", linewidth=1.2)

    ax_hist.set_title("Concatenated 1D CIELab Feature Vector (Total D = 136)", fontsize=12, fontweight="bold", pad=10)
    ax_hist.set_xlabel("Descriptor Bin Index", fontweight="bold")
    ax_hist.set_ylabel("L1 Normalized Density", fontweight="bold")
    ax_hist.set_xlim(-1, len(desc))
    ax_hist.legend(loc="upper right", frameon=True)

    out_path = out_dir / "example_descriptor_cielab.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


# -----------------------------------------------------------------------------
# 2. HSV Descriptor (16, 16, 8) with Hue Valid Weighting
# -----------------------------------------------------------------------------
def plot_hsv_example(img_bgr, img_name, out_dir, s_min=20, v_min=40):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    bins = (16, 16, 8)

    # OpenCV Hue is in [0, 180), S and V in [0, 256)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Valid chromatic pixels mask
    valid_mask = (S >= s_min) & (V >= v_min)
    chromatic_weight = float(np.mean(valid_mask))

    h_H = cv2.calcHist([H], [0], valid_mask.astype(np.uint8), [bins[0]], [0, 180]).flatten()
    if h_H.sum() > 0:
        h_H = (h_H / h_H.sum()) * chromatic_weight

    h_S = cv2.calcHist([S], [0], None, [bins[1]], [0, 256]).flatten()
    if h_S.sum() > 0:
        h_S /= h_S.sum()

    h_V = cv2.calcHist([V], [0], None, [bins[2]], [0, 256]).flatten()
    if h_V.sum() > 0:
        h_V /= h_V.sum()

    desc = np.concatenate([h_H, h_S, h_V]).astype(np.float64)
    desc /= desc.sum()  # L1 normalize

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.2, 0.05])

    # Left: Original Image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax_img.set_title(f"Query Image\n({img_name})", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Right: 1D Concatenated Descriptor
    ax_hist = fig.add_subplot(gs[0, 1])
    x = np.arange(len(desc))

    # Segment slices
    i_H = bins[0]
    i_S = i_H + bins[1]

    ax_hist.bar(x[:i_H], desc[:i_H], color="#E41A1C", label=f"H: Hue ({bins[0]} bins, weighted)", width=0.85, edgecolor="black", linewidth=0.3)
    ax_hist.bar(x[i_H:i_S], desc[i_H:i_S], color="#377EB8", label=f"S: Saturation ({bins[1]} bins)", width=0.85, edgecolor="black", linewidth=0.3)
    ax_hist.bar(x[i_S:], desc[i_S:], color="#984EA3", label=f"V: Value ({bins[2]} bins)", width=0.85, edgecolor="black", linewidth=0.3)

    # Dividing lines
    ax_hist.axvline(i_H - 0.5, color="#333333", linestyle="--", linewidth=1.2)
    ax_hist.axvline(i_S - 0.5, color="#333333", linestyle="--", linewidth=1.2)

    ax_hist.set_title(f"Concatenated 1D HSV Feature Vector (Total D = {len(desc)})", fontsize=12, fontweight="bold", pad=10)
    ax_hist.set_xlabel("Descriptor Bin Index", fontweight="bold")
    ax_hist.set_ylabel("L1 Normalized Density", fontweight="bold")
    ax_hist.set_xlim(-1, len(desc))
    ax_hist.legend(loc="upper right", frameon=True)

    out_path = out_dir / "example_descriptor_hsv.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    img_path = get_sample_image()
    print(f"Generating descriptor plots using: {img_path}")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")

    plot_cielab_example(img_bgr, img_path.name, PLOTS_DIR)
    plot_hsv_example(img_bgr, img_path.name, PLOTS_DIR)


if __name__ == "__main__":
    main()