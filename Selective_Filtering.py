import cv2
import numpy as np

# Load image
img = cv2.imread('lena;.jpg')

# Split the image into 3 channels
b, g, r = cv2.split(img)

# Define the filter kernel size
kernel_size = (15, 15)

# Apply Gaussian Blur to each channel
b_filter = cv2.GaussianBlur(b, kernel_size, 0)
g_filter = cv2.GaussianBlur(g, kernel_size, 0)
r_filter = cv2.GaussianBlur(r, kernel_size, 0)

# Compute the difference between original and filtered images
b_diff = b - b_filter
g_diff = g - g_filter
r_diff = r - r_filter

# Add the difference to the original image
b_selective = cv2.add(b, b_diff)
g_selective = cv2.add(g, g_diff)
r_selective = cv2.add(r, r_diff)

# Merge the channels back together
selective_img = cv2.merge((b_selective, g_selective, r_selective))

# Display the result
cv2.imshow('Original Image', img)
cv2.imshow('Selective Filtering', selective_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

