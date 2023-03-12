import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the 2D DFT of the image
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the low frequency components to the center
dft_shift = np.fft.fftshift(dft)

# Compute the magnitude spectrum
magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Plot the original image and its magnitude spectrum
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img, cmap='gray')
ax1.set_title('Input Image')
ax1.set_xticks([])
ax1.set_yticks([])
ax2.imshow(magnitude_spectrum, cmap='gray')
ax2.set_title('Magnitude Spectrum')
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()
