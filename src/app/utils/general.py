import pickle
import json
from typing import Tuple
import app.utils.image as I
import numpy as np
import torch
import app as app

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

def xywh2xyxy(coords: np.ndarray) -> np.ndarray:
    """Transform coordinates from xywh, (top-left, width and height)
    to xyxy, (top-left, bottom-right)
    Args:
        coords (np.ndarray): np.ndarray with xywh coordinates
    Returns:
        np.ndarray: numpy array with xyxy coordinates
    """
    new_coords = np.zeros_like(coords)
    new_coords[0] = coords[0]
    new_coords[1] = coords[1]
    new_coords[2] = coords[0] + coords[2]
    new_coords[3] = coords[1] + coords[3]
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

def get_dimensions_box(coords: np.ndarray) -> Tuple[int, int]:
    """Get Height and Width dimensions from xyxy coordinates
    Args:
        coords (np.ndarray): box coordinates in xyxy format
    Returns:
        Tuple[int, int]: Height and width
    """
    w, h = (np.abs(coords[0] - coords[2]), 
            np.abs(coords[1] - coords[3]))
    return int(h), int(w)

def set_model_in_cache(**numpy_arrays_dict):
    """Cache numpy arrays in a dict
    """
    db = list(app.get_redis_db())[0]
    for key, numpy_array in numpy_arrays_dict.items():
        # caching numpy arrays models
        db.set(key, pickle.dumps(numpy_array))
        
def get_model_from_cache(*key_names) -> dict:
    """get models from cache into a dict with the models arrays
    Returns:
        dict: dict with the models arrays from redis db
    """
    db = list(app.get_redis_db())[0]
    models = {}
    for key in key_names:
        models[key] = pickle.loads(db.get(key))
    return models

def set_blend_img_in_cache(image: torch.Tensor) -> None:
    """set blend image in cache db
    Args:
        image (torch.Tensor): image tensor
    """
    db = list(app.get_redis_db())[0]
    db.set('b64resized_blend',
           json.dumps(I._pil_to_base64(I._to_pil(image))))