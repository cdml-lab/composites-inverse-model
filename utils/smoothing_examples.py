import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.signal import savgol_filter
from skimage.restoration import denoise_tv_chambolle
from skimage import io, img_as_float
from pathlib import Path
import cv2

# Load example image
base_path = Path.cwd()
img_path = base_path / "plots/example.jpg"
image = img_as_float(io.imread(img_path, as_gray=True))  # Ensure it's grayscale float

# Define output directory
output_dir = base_path
assert image.shape == (40, 40), "Expected image size of 40x40 pixels."

# Smoothing methods
sigma = 1.0
results = {}


# Original
results['original'] = (image, f"Original Image")

# Gaussian
gaussian = gaussian_filter(image, sigma=sigma)
results['gaussian'] = (gaussian, f"Gaussian (σ={sigma})")

# Uniform
size = int(2 * sigma + 1)
uniform = uniform_filter(image, size=size)
results['uniform'] = (uniform, f"Uniform (size={size})")

# Median
median = median_filter(image, size=size)
results['median'] = (median, f"Median (size={size})")

# Savitzky-Golay
window = int(2 * sigma + 1)
if window % 2 == 0:
    window += 1
savgol = savgol_filter(image, window_length=window, polyorder=2, axis=0)
savgol = savgol_filter(savgol, window_length=window, polyorder=2, axis=1)
results['savgol'] = (savgol, f"Savitzky-Golay (window={window}, polyorder=2)")

# Bilateral
Z_norm = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
Z_8bit = np.uint8(Z_norm)
bilateral = cv2.bilateralFilter(Z_8bit, d=9, sigmaColor=75, sigmaSpace=sigma)
bilateral = bilateral.astype(float) / 255 * (np.max(image) - np.min(image)) + np.min(image)
results['bilateral'] = (bilateral, f"Bilateral (σ_space={sigma}, σ_color=75)")

# Anisotropic diffusion (TV denoising)
tv = denoise_tv_chambolle(image, weight=sigma)
results['anisotropic'] = (tv, f"Anisotropic Diffusion (weight={sigma})")



# Plot and save
for name, (img, title) in results.items():
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='plasma')
    plt.title(f"{name.capitalize()} Smoothing\n{title}")
    plt.axis('off')
    save_path = output_dir / f"plots/{name}_smoothing_sigma{sigma}.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

"Saved all smoothing visualizations."
