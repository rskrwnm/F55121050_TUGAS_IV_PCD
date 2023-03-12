import cv2
import numpy as np

def median_filter(data, kernel_size):
    temp = []
    indexer = kernel_size // 2
    data_final = np.zeros((len(data), len(data[0])))

    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(kernel_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(kernel_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(kernel_size):
                            temp.append(data[i + z - indexer][j + k - indexer])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []

    return data_final

# membaca gambar dengan opencv
img = cv2.imread('lena_gray.jpeg', 0)

# menambahkan gaussian noise dengan numpy
noisy_img_1 = img + np.random.normal(0, 10, img.shape)
noisy_img_2 = img + np.random.normal(0, 20, img.shape)
noisy_img_3 = img + np.random.normal(0, 30, img.shape)

# filter median dengan ukuran kernel 3
filtered_img_1 = median_filter(noisy_img_1, 3)
filtered_img_2 = median_filter(noisy_img_2, 3)
filtered_img_3 = median_filter(noisy_img_3, 3)

# menampilkan gambar asli dan hasil filtering
cv2.imshow('Original Image', img)
cv2.imshow('Noisy Image 1', noisy_img_1.astype(np.uint8))
cv2.imshow('Filtered Image 1', filtered_img_1.astype(np.uint8))
cv2.imshow('Noisy Image 2', noisy_img_2.astype(np.uint8))
cv2.imshow('Filtered Image 2', filtered_img_2.astype(np.uint8))
cv2.imshow('Noisy Image 3', noisy_img_3.astype(np.uint8))
cv2.imshow('Filtered Image 3', filtered_img_3.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
