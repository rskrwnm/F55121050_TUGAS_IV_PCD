import cv2
import numpy as np

# Load the image
img = cv2.imread('img1.jpg')

# Apply Gaussian blur
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)

# Apply unsharp masking
unsharp_mask = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

# Display the results
cv2.imshow('Original', img)
cv2.imshow('Unsharp Masking', unsharp_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
