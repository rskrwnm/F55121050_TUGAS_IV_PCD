import cv2

def minimumBoxFilter(n, path_to_image):
  # Load image using OpenCV
  img = cv2.imread(path_to_image)

  # Creates the shape of the kernel
  size = (n, n)
  shape = cv2.MORPH_RECT
  kernel = cv2.getStructuringElement(shape, size)

  # Applies the minimum filter with kernel NxN using OpenCV's erode function
  imgResult = cv2.erode(img, kernel)

  # Shows the result using OpenCV's imshow function
  cv2.namedWindow('Result with n ' + str(n), cv2.WINDOW_NORMAL) # Adjust the window length
  cv2.imshow('Result with n ' + str(n), imgResult)

# Test the function with different kernel sizes
if __name__ == "__main__":
  path_to_image = 'img1.jpg'
  minimumBoxFilter(3, path_to_image)
  minimumBoxFilter(5, path_to_image)
  minimumBoxFilter(7, path_to_image)
  minimumBoxFilter(11, path_to_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()
