import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('lena.jpg', 0)

# Calculate the DFT of the image
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)


# Gaussian Lowpass Filter
sigma = 40
rows, cols = img.shape
crow, ccol = rows//2, cols//2
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
glp = np.exp(-((X-ccol)**2 + (Y-crow)**2) / (2*sigma**2))
glp_shift = np.fft.ifftshift(glp)
gaussian_lp = dft_shift * glp_shift
gaussian_lp_shift = np.fft.ifftshift(gaussian_lp)
gaussian_lp_img = np.fft.ifft2(gaussian_lp_shift)
gaussian_lp_img = np.abs(gaussian_lp_img)

# Ideal Highpass Filter
mask = np.ones((rows, cols), np.uint8)
r = 80
cv2.circle(mask, (ccol, crow), r, 0, -1)
ideal_hp = dft_shift * mask
ideal_hp_shift = np.fft.ifftshift(ideal_hp)
ideal_hp_img = np.fft.ifft2(ideal_hp_shift)
ideal_hp_img = np.abs(ideal_hp_img)

# Display the results
plt.subplot(2, 3, 1), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(gaussian_lp_img)
plt.title('Gaussian Lowpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4), plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(ideal_hp_img)
plt.title('Ideal Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
