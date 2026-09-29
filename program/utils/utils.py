import cv2

def load_image(img_path):
    """
    Load an image from the specified path and convert it to RGB format.

    Args:
        img_path (str): The path to the image file.
    
    Returns:
        numpy.ndarray: The loaded image in RGB format.
    """
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def resize_image(image, target_size):
    """
    Resize an image to the specified target size.

    Args:
        image (numpy.ndarray): The input image.
        target_size (tuple): The target size as (width, height).

    Returns:
        numpy.ndarray: The resized image.
    """
    return cv2.resize(image, target_size)

# Ukuran Image 512x512
def get_point_prompt(fold_number):
    """
    Get the point prompt for the specified fold number.

    Args:
        fold_number (int): The fold number (1 to 5).
    """
    points = [
                [250, 173], # Untuk Fold 1
                [249, 175], # Untuk Fold 2
                [252, 173], # Untuk Fold 3
                [251, 173], # Untuk Fold 4
                [247, 176], # Untuk Fold 5
            ]
    return points[fold_number - 1] if 1 <= fold_number <= 5 else None