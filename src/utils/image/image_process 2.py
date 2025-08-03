import cv2
import numpy as np

def image_resize(path: str, destination: str, new_height: int, new_width: int) -> None:
    """ Resizes image to new height/width
    Args:
        str: os path to the image
        str: os destination for the image
        int: new hieght after resize
        int: new width after resize
    Return:
        None
    """
    image = cv2.imread(path)
    resized_image = cv2.resize(image, (new_height, new_width), interpolation = cv2.INTER_AREA)
    cv2.imwrite(destination, resized_image)

def image_normalize(path: str, destination: str):
    """ Normalizes image pixels to new height/width
    Args:
        str: os path to the image
        str: os destination for the image
        int: new hieght after resize
        int: new width after resize
    Return:
        None
    """
    # Load the image
    image = cv2.imread(path)
    normalized_image = np.zeros_like(image, dtype=np.float32)
    # Normalize the image to the range [0, 1]
    # NORM_MINMAX ensures values are scaled between the specified alpha and beta
    # In this case, alpha=0, beta=1
    cv2.normalize(image, normalized_image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite(destination, normalized_image)
    