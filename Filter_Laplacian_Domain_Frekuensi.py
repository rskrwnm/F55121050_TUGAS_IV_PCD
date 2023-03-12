import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('img1.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Laplacian filter
laplacian_img = cv2.Laplacian(gray, cv2.CV_64F)

# Convert the image to uint8 and normalize the pixel values to [0, 255]
laplacian_img = cv2.convertScaleAbs(laplacian_img)

# Display the results
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(laplacian_img, cmap='gray')
plt.title('Laplacian Filter'), plt.xticks([]), plt.yticks([])
plt.show()
