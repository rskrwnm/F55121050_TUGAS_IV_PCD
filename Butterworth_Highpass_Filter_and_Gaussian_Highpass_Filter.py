import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image in grayscale
img = cv2.imread('img1.jpg', 0)

# Calculate the DFT of the image
dft = np.fft.fft2(img)
dft_shift = np.fft.fftshift(dft)

# Butterworth Highpass Filter
n = 4
d0 = 80
rows, cols = img.shape
crow, ccol = rows//2, cols//2
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
dist = np.sqrt((X-ccol)**2 + (Y-crow)**2)
bhpf = 1 / (1 + (d0/dist)**(2*n))
bhpf_shift = np.fft.ifftshift(bhpf)
butterworth_hp = dft_shift * bhpf_shift
butterworth_hp_shift = np.fft.ifftshift(butterworth_hp)
butterworth_hp_img = np.fft.ifft2(butterworth_hp_shift)
butterworth_hp_img = np.abs(butterworth_hp_img)

# Gaussian Highpass Filter
sigma = 30
ghpf = 1 - np.exp(-((X-ccol)**2 + (Y-crow)**2) / (2*sigma**2))
ghpf_shift = np.fft.ifftshift(ghpf)
gaussian_hp = dft_shift * ghpf_shift
gaussian_hp_shift = np.fft.ifftshift(gaussian_hp)
gaussian_hp_img = np.fft.ifft2(gaussian_hp_shift)
gaussian_hp_img = np.abs(gaussian_hp_img)

# Display the results
plt.subplot(2, 3, 1), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(butterworth_hp_img, cmap='gray')
plt.title('Butterworth Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(gaussian_hp_img, cmap='gray')
plt.title('Gaussian Highpass Filter'), plt.xticks([]), plt.yticks([])
plt.show()
