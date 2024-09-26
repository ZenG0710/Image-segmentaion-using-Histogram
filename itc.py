
from skimage import io, color
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_ubyte, img_as_float
from scipy import ndimage as nd
from skimage.filters import sobel
from skimage.measure import shannon_entropy
from scipy.stats import kurtosis, skew
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray

def calculate_image_parameters(image, original_image=None):
    float_image = img_as_float(image)
    
    # Mean, Standard deviation, Contrast
    mean_intensity = np.mean(float_image)
    std_dev = np.std(float_image)
    contrast = np.max(float_image) - np.min(float_image)
    
    # Entropy
    entropy_value = shannon_entropy(float_image)
    
    # Sobel edge detection for sharpness/edges
    edge_sobel = sobel(float_image)
    sharpness = np.mean(edge_sobel)
    
    # Skewness and Kurtosis
    skewness_value = skew(float_image.flat)
    kurtosis_value = kurtosis(float_image.flat)
    
    # Variance
    variance_value = np.var(float_image)
    
    # PSNR (Peak Signal-to-Noise Ratio) - Ensure both images are grayscale
    if original_image is not None:
        if original_image.ndim == 3:
            original_image_gray = rgb2gray(original_image)
        else:
            original_image_gray = original_image
        
        if float_image.ndim == 3:
            float_image_gray = rgb2gray(float_image)
        else:
            float_image_gray = float_image
            
        psnr_value = psnr(original_image_gray, float_image_gray)
    else:
        psnr_value = None

    # Collect and return all parameters
    return {
        "Mean Intensity": mean_intensity,
        "Standard Deviation": std_dev,
        "Contrast": contrast,
        "Entropy": entropy_value,
        "Sharpness (Edge Intensity)": sharpness,
        "Variance": variance_value,
        "Skewness": skewness_value,
        "Kurtosis": kurtosis_value,
        "PSNR": psnr_value
    }

def load_image(image_path):
    img = io.imread(image_path)
    return img

def convert_to_float(img):
    if img.ndim == 3:
        img = color.rgb2gray(img)  # Convert RGB to grayscale
    return img_as_float(img)

def denoise_image(float_img):
    sigma_est = np.mean(estimate_sigma(float_img))
    denoise_img = denoise_nl_means(float_img, h=1.15 * sigma_est, fast_mode=False, patch_size=5, patch_distance=3)
    return denoise_img

def convert_to_ubyte(denoise_img):
    return img_as_ubyte(denoise_img)

def plot_histogram(image, bins=100, range_values=(0, 255)):
    plt.hist(image.flat, bins=bins, range=range_values)
    plt.title("Histogram of Denoised Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()

def segment_image(image):
    segm1 = (image <= 57)
    segm2 = (image > 57) & (image <= 110)
    segm3 = (image > 110) & (image <= 210)
    segm4 = (image > 210)
    return segm1, segm2, segm3, segm4

def create_segmented_image(shape, segm1, segm2, segm3, segm4):
    all_segments = np.zeros((shape[0], shape[1], 3))
    all_segments[segm1] = (1, 0, 0)
    all_segments[segm2] = (0, 1, 0)
    all_segments[segm3] = (0, 0, 1)
    all_segments[segm4] = (1, 1, 0)
    return all_segments

def apply_morphological_operations(segm):
    opened = nd.binary_opening(segm, np.ones((3, 3)))
    closed = nd.binary_closing(opened, np.ones((3, 3)))
    return closed

def create_cleaned_image(shape, segm1, segm2, segm3, segm4):
    all_segments_cleaned = np.zeros((shape[0], shape[1], 3))
    all_segments_cleaned[segm1] = (1, 0, 0)
    all_segments_cleaned[segm2] = (0, 1, 0)
    all_segments_cleaned[segm3] = (0, 0, 1)
    all_segments_cleaned[segm4] = (1, 1, 0)
    return all_segments_cleaned

def additional_processing(image):
    rotated = nd.rotate(image, 45, reshape=False)
    shifted = nd.shift(rotated, shift=(5, 5, 0))
    zoomed = nd.zoom(shifted, (1.2, 1.2, 1))
    filtered = nd.median_filter(zoomed, size=3)
    edges = nd.sobel(filtered[:, :, 0])
    zoomed_back = nd.zoom(filtered, (0.8, 0.8, 1))
    edge_stack = np.stack([edges, edges, edges], axis=2)
    combined = np.clip(filtered + edge_stack, 0, 1)
    blurred = nd.gaussian_filter(combined, sigma=(2, 2, 0))
    return blurred

def plot_images(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    image_path = r"C:\Users\ASUS\Pictures\composition-beautiful-flowers-wallpaper.jpg"
    
    # Load and process image
    img = load_image(image_path)
    float_img = convert_to_float(img)
    denoise_img = denoise_image(float_img)
    denoise_img_as_8byte = convert_to_ubyte(denoise_img)
    
    # Calculate image parameters for the denoised image
    parameters = calculate_image_parameters(denoise_img_as_8byte, img)
    
    # Print calculated parameters
    for param, value in parameters.items():
        print(f"{param}: {value}")
    
    # Continue with further processing
    plot_histogram(denoise_img_as_8byte)
    
    segm1, segm2, segm3, segm4 = segment_image(denoise_img_as_8byte)
    all_segments = create_segmented_image(denoise_img_as_8byte.shape, segm1, segm2, segm3, segm4)
    
    segm1_cleaned = apply_morphological_operations(segm1)
    segm2_cleaned = apply_morphological_operations(segm2)
    segm3_cleaned = apply_morphological_operations(segm3)
    segm4_cleaned = apply_morphological_operations(segm4)
    
    all_segments_cleaned = create_cleaned_image(denoise_img_as_8byte.shape, segm1_cleaned, segm2_cleaned, segm3_cleaned, segm4_cleaned)
    
    further_processed = additional_processing(all_segments_cleaned)
    
    plot_images([img, denoise_img_as_8byte, all_segments, all_segments_cleaned, further_processed], 
                ['Initial Image', 'Denoised Image', 'Segmented Image', 'Cleaned Segments', 'Further Processed Image'])

if __name__ == "__main__":
    main()
