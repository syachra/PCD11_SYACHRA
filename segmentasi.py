import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def sobel_edge_detection(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    gradient_x = ndimage.convolve(image, sobel_x)
    gradient_y = ndimage.convolve(image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)

    return gradient_magnitude

def threshold_segmentation(image, threshold_value):
    binary_image = np.zeros_like(image)
    binary_image[image >= threshold_value] = 255
    return binary_image

# Baca gambar
image = imageio.imread('//content//sereal.jpg')

# Deteksi tepi menggunakan Sobel
edge_image = sobel_edge_detection(image)

# Lakukan thresholding dengan empat nilai threshold yang berbeda
threshold_values = [30, 60, 90, 120]
threshold_results = []

for threshold in threshold_values:
    segmented_image = threshold_segmentation(edge_image, threshold)
    threshold_results.append(segmented_image)

# Tampilkan hasil dengan layout 2x3
plt.figure(figsize=(15, 10))

# Subplot untuk gambar asli
plt.subplot(2, 3, 1)  # 2 baris, 3 kolom, posisi 1
plt.imshow(image)
plt.title('Gambar Asli')
plt.axis('off')

# Subplot untuk hasil deteksi tepi
plt.subplot(2, 3, 2)  # 2 baris, 3 kolom, posisi 2
plt.imshow(edge_image, cmap='gray')
plt.title('Deteksi Tepi Sobel')
plt.axis('off')

# Subplot untuk hasil threshold
for i, (thresh, result) in enumerate(zip(threshold_values, threshold_results)):
    plt.subplot(2, 3, i + 3)  # 2 baris, 3 kolom, posisi 3,4,5,6
    plt.imshow(result, cmap='gray')
    plt.title(f'Threshold = {thresh}')
    plt.axis('off')

plt.tight_layout()
plt.show()
