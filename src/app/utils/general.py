import numpy as np

def xyxy2xywh(coords: np.ndarray) -> np.ndarray:
    """Transform coordinates from xyxy, (top-left, bottom-right)
    to xywh, (top-left, width and height)
    Args:
        coords (np.ndarray): np.ndarray with xyxy coordinates

    Returns:
        np.ndarray: Numpy array with xywh coordinates
    """
    new_coords = np.zeros_like(coords)
    new_coords[0] = coords[0]
    new_coords[1] = coords[1]
    new_coords[2] = np.abs(coords[0] - coords[2])
    new_coords[3] = np.abs(coords[1] - coords[3])
    return new_coords

def scale_coordinates(new_shape: tuple, coordinates: np.ndarray,
                      old_shape: tuple) -> np.ndarray:
    """ Scale coordinates from old_shape image to new_shape image
    Args:
        new_shape (tuple): (H, W) new image shape
        coordinates (np.ndarray): box coordinates xyxy
        old_shape (tuple): (H, W) old image shape
    Returns:
        np.ndarray: 
    """
    if coordinates.dtype != np.float32:
        coordinates = coordinates.astype(np.float32)
    scale_coords:np.ndarray = coordinates.copy()
    r = min(old_shape[0]/new_shape[0],
            old_shape[1]/new_shape[1])
    unpad = (round(new_shape[0] * r),
             round(new_shape[1] * r))
    dw, dh = ((old_shape[1] - unpad[1])//2,
              (old_shape[0] - unpad[0])//2)
    # Apply pad
    scale_coords[[0, 2]] -= dw # x padding
    scale_coords[[1, 3]] -= dh # y padding
    # scale
    scale_coords /= r
    return scale_coords.round().astype(np.int32)