import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_lowpass_filter(image, cutoff):
    # Menghitung transformasi Fourier dari citra
    f = np.fft.fft2(image)

    # Menggeser nol frekuensi ke tengah citra
    fshift = np.fft.fftshift(f)

    # Membuat filter lowpass ideal
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= cutoff:
                mask[i, j] = 1

    # Mengalikan filter dengan transformasi Fourier
    fshift_filtered = fshift * mask

    # Menggeser kembali nol frekuensi ke sudut kiri atas citra
    f_filtered = np.fft.ifftshift(fshift_filtered)

    # Menghitung transformasi Fourier balik
    image_filtered = np.fft.ifft2(f_filtered)
    image_filtered = np.abs(image_filtered)

    return image_filtered

def butterworth_lowpass_filter(image, cutoff, n):
    # Menghitung transformasi Fourier dari citra
    f = np.fft.fft2(image)

    # Menggeser nol frekuensi ke tengah citra
    fshift = np.fft.fftshift(f)

    # Membuat filter lowpass Butterworth
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = 1 / (1 + (d / cutoff) ** (2 * n))

    # Mengalikan filter dengan transformasi Fourier
    fshift_filtered = fshift * mask

    # Menggeser kembali nol frekuensi ke sudut kiri atas citra
    f_filtered = np.fft.ifftshift(fshift_filtered)

    # Menghitung transformasi Fourier balik
    image_filtered = np.fft.ifft2(f_filtered)
    image_filtered = np.abs(image_filtered)

    return image_filtered

# Membaca gambar
img = cv2.imread('img1.jpg', 0)

# Mengaplikasikan Ideal Lowpass Filter pada gambar
img_ideal = ideal_lowpass_filter(img, 50)

# Mengaplikasikan Butterworth Lowpass Filter pada gambar
img_butterworth = butterworth_lowpass_filter(img, 50, 2)

# Menampilkan gambar asli dan hasil filter
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_ideal, cmap='gray'), plt.title('Ideal LPF')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_butterworth, cmap='gray'), plt.title('Butterworth LPF')
plt.xticks([]), plt.yticks([])
plt.show()
