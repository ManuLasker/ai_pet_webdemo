import torch
import cv2
import pickle
import numpy as np
from redis import Redis
from app.utils.general import *
from typing import Tuple, Union, overload
from io import BytesIO
from base64 import b64decode, b64encode
from PIL import Image
from app import Predictor
from app.blending.utils import *
from torchvision import transforms as T

def _base64_to_pil(base64_image:str, convert_to_mask: bool = False) -> Image.Image:
    """Convert base64 image string to PIL image
    Args:
        base64Image (str): base 64 str image decode
    Returns:
        Image.Image: Pil image
    """
    # Select just the image information if there is more information
    if len(base64_image.split(",")) > 1:
        _, base64_image = base64_image.split(",")
    pil_image = Image.open(BytesIO(b64decode(base64_image)))
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    if convert_to_mask:
        pil_image = pil_image.convert("L")
    return pil_image

def _pil_to_base64(pil_image: Image.Image) -> str:
    """Convert pil image to base64 encode string format
    Args:
        pil_image (Image.Image): pil Image
    Returns:
        str: string base64 image
    """
    _buffer = BytesIO()
    if pil_image.mode != "RGBA":
        pil_image.save(_buffer, format="JPEG")
    else:
        pil_image.save(_buffer, format="PNG")
    img_str = b64encode(_buffer.getvalue()).decode("utf-8")
    return img_str

def _to_tensor(image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
    """Convert pil Image or numpy Image to tensor Image
    Args:
        image (Union[Image.Image, np.ndarray]): pil Image
    Returns:
        torch.Tensor: tensor Image
    """
    return T.ToTensor()(image)

def _to_numpy(image: Union[Image.Image, torch.Tensor]) -> np.ndarray:
    """Convert Pil Image or Tensor Image to numpy Image
    Args:
        image (Union[Image.Image, torch.Tensor]): Pil Image or Tensor Image
    Returns:
        np.ndarray: numpy Image
    """
    if isinstance(image, torch.Tensor):
        image = np.transpose(image.cpu().numpy(),
                             (1, 2, 0))
        return image
    elif isinstance(image, Image.Image):
        return np.array(image)

def _to_pil(image: Union[torch.Tensor, np.ndarray]) -> Image.Image:
    """Convert tensor Image or numpy array Image to PIL image format
    Args:
        image (Union[torch.Tensor, np.ndarray]): tensor Image or numpy Image
    Returns:
        Image.Image: PIL image format
    """
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        return T.ToPILImage()(image)
    
def get_image_base64_shape(base64_image: str) -> tuple:
    """Given a base64 image format return Height and Width
    Args:
        base64_image (str): bas64 string encode image
    Returns:
        tuple: (H, W) tuple size for image
    """
    return _base64_to_pil(base64_image).size[::-1]

def resize_with_pad(image: Image.Image, new_shape=(224, 224)) -> Image.Image:
    """Receive a pil Image and resize applying padding, this method also remove padding,
    and make the inverse method
    Args:
        image (Image.Image): pil Image
        new_shape (tuple, optional): Shape you want to resize. Defaults to (224, 224).
    Returns:
        Image.Image: pil Image resize
    """
    # ratio
    old_w, old_h = image.size # PIL image return (W, H) up small size
    new_h, new_w = new_shape # new_shape (H, W) down high size
    r = min(new_h/old_h, new_w/old_w)
    
    # up or down resize
    if r > 1: 
        # activate apply resize up inverse eliminating pad
        up = True
        r = 1/max(new_h/old_h, new_w/old_w) 
        # Get pad to Eliminate it
        new_unpad_h, new_unpad_w = round(new_h * r), round(new_w * r)
        dh, dw = (old_h - new_unpad_h)/2, (old_w - new_unpad_w)/2
    else:
        # is resize down
        up = False
        # Get padding
        new_unpad_h, new_unpad_w = round(old_h * r), round(old_w * r)
        dh, dw = (new_h - new_unpad_h)/2, (new_w - new_unpad_w)/2
        
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    
    if up:
        image = _to_tensor(image) # ch x H x W
        # delete pad
        image = image[:, top: bottom + new_unpad_h, left: right + new_unpad_w]
        image = _to_pil(image) # transform to pil again
        transform = T.Resize(size=(new_h, new_w))
    else:
        transform = T.Compose([
            T.Resize(size=(new_unpad_h, new_unpad_w)),
            T.Pad(padding=(left, top, right, bottom))
        ])
    # resize image
    return transform(image)


def apply_grabcut_rect(image: np.ndarray, xywh_box: np.ndarray)->np.ndarray:
    """Apply grabcut algorithm with rect mode
    Args:
        image (np.ndarray): numpy array uint8
        xywh_box (np.ndarray): xywh box separator
    Returns:
        np.ndarray: mask
    """
    # validate dtype
    if (image.dtype != np.uint8):
        image = image * 255 if image.max() <= 1 else image
        image = image.astype(np.uint8)
    # Create mask
    mask:np.ndarray = np.zeros(image.shape[:2], dtype=np.uint8)
    # Create parameters for grabcut
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    # Apply grabcut for init with rect
    cv2.grabCut(image, mask, xywh_box, bgdModel=bgdModel,
                fgdModel=fgdModel, iterCount=5,
                mode=cv2.GC_INIT_WITH_RECT)
    # Cache bgdModel and fgdModel
    set_model_in_cache(fgdModel=fgdModel, bgdModel=bgdModel, mask=mask)
    # Get mask_grabcut
    mask_grabcut = np.where((mask==2)|(mask==0), 0, 1)
    return inv_preprocess_numpy_image(mask_grabcut)

def apply_grabcut_mask(image: np.ndarray,
                       mask_paint:np.ndarray) -> np.ndarray:
    """apply grabcut algorithm in mask mode
    Args:
        image (np.ndarray): numpy image array uint8
        mask_paint (np.ndarray): numpy image array uint8,
        is the paint from canvas tool
    Returns:
        np.ndarray: mask
    """
    # Get fgdModel and bgdModel
    fgdModel, bgdModel, mask_helper_square = get_model_from_cache('fgdModel',
                                                                  'bgdModel',
                                                                  'mask').values()
    # get information from mask_paint to mask_helper_square
    mask = mask_helper_square.copy()
    mask[mask_paint == 255] = 1
    mask[mask_paint == 0] = 0

    # Apply grabcut
    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel=bgdModel,
                                           fgdModel=fgdModel, iterCount=5,
                                           mode=cv2.GC_INIT_WITH_MASK)
    
    # Get mask_grabcut
    mask_grabcut = np.where((mask==2)|(mask==0), 0, 1)
    return inv_preprocess_numpy_image(mask_grabcut)

def get_grabcut_by_mask(base64_image: str, b64mask_paint:str, new_size: list,
                        old_size: list) -> Tuple[str, str, str]:
    """get grab cut mask using mask_helper from square method
    Args:
        base64_image (str): base64 image encode
        b64mask_paint (str): base64 mask paint helper
        new_size (list): original size of the image
        old_size (list): resize size of the new image
    Returns:
        Tuple[str, str, str]: return base64 masks for the original 
        and the resized and the preview
    """
    # load to numpy
    original_numpy_image = _to_numpy(_base64_to_pil(base64_image))
    # load b64 mask paint to numpy this is in 224x224 we need to resize up
    mask_paint_helper = _base64_to_pil(b64mask_paint, convert_to_mask=True)
    mask_paint_helper = _to_numpy(resize_with_pad(mask_paint_helper, new_size))
    # apply grab cut
    original_mask_numpy = apply_grabcut_mask(image=original_numpy_image,
                       mask_paint=mask_paint_helper)
    # to pil
    original_mask_pil = _to_pil(original_mask_numpy)
    # resize pad
    resized_mask_pil = resize_with_pad(original_mask_pil, old_size)
    # get preview
    original_preview_cut = get_preview_cut(original_numpy_image, original_mask_numpy)
    original_preview_cut = _to_pil(original_preview_cut)
    resize_preview_cut = resize_with_pad(original_preview_cut, old_size)
    
    return (_pil_to_base64(original_mask_pil),
            _pil_to_base64(resized_mask_pil),
            _pil_to_base64(resize_preview_cut))
    
def get_grabcut_by_rect(base64_image: str, resized_xyxy_box: list, 
                         new_size: list, old_size: list) -> Tuple[str, str, str]:
    """get grab cut mask using rect xyxy dimensions, 
    the xyxy dimensions are for the resized image, we need to reescale the coordinates
    Args:
        base64_image (str): base64 image encode 
        resized_xyxy_box (list): list [x0, y0, x1, y1] for the resized size
        new_size (list): original size of the image
        old_size (list): resize size of the new image
    Returns:
        Tuple[str, str]: return base64 masks for the original
        and the resized and the preview
    """
    # get numpy image
    original_numpy_image = _to_numpy(_base64_to_pil(base64_image))
    # get coordinates to numpy
    resized_xyxy_box = np.array(resized_xyxy_box, dtype=np.int32)
    # reescale coordinates
    original_xyxy_box = scale_coordinates(new_shape=new_size,
                                          coordinates=resized_xyxy_box,
                                          old_shape=old_size)
    # Change coordinates xyxy to xywh format
    original_xywh_box = xyxy2xywh(original_xyxy_box)
    # Apply grabcut
    original_numpy_mask = apply_grabcut_rect(original_numpy_image, original_xywh_box)
    # Transform to pil image
    original_pil_mask = _to_pil(original_numpy_mask)
    # resize original mask
    resized_pil_mask = resize_with_pad(original_pil_mask, old_size)
    # get preview cut image
    original_preview_cut = get_preview_cut(original_numpy_image, original_numpy_mask)
    # Transform to pil image
    original_preview_cut = _to_pil(original_preview_cut)
    # resize preview
    resized_preview_cut = resize_with_pad(original_preview_cut, old_size)
    # transform to base 64 
    return (_pil_to_base64(original_pil_mask),
            _pil_to_base64(resized_pil_mask),
            _pil_to_base64(resized_preview_cut))

def preprocess_numpy_image(image: np.ndarray) -> np.ndarray:
    """preprocess np.uint8 image or np.float32 if it not normalize
    Args:
        image (np.ndarray): numpy image array
    Returns:
        np.ndarray: numpy image array preprocess
    """
    image = image.astype(np.float32) if image.dtype == np.uint8 else image
    image = image/255.0 if image.max() > 1 else image
    return image

def inv_preprocess_numpy_image(image: np.ndarray) -> np.ndarray:
    """inverse preprocess numpy image and return a np.uint8 numpy image array
    Args:
        image (np.ndarray): numpy image
    Returns:
        np.ndarray: inverse preprocess numpy image
    """
    image = (image * 255.0).astype(np.uint8)
    return image

def get_preview_cut(image:np.ndarray,
                    mask:np.ndarray) -> np.ndarray:
    """Get preview cut from image using the mask, formula:
    preview_cut = image * mask
    Args:
        image (np.ndarray): image numpy
        mask (np.ndarray): mask numpy
    Returns:
        np.ndarray: preview cut
    """
    image = preprocess_numpy_image(image)
    mask = preprocess_numpy_image(mask)[:, :, np.newaxis]
    preview_cut = image * mask
    return inv_preprocess_numpy_image(preview_cut)

def predict_mask_using_model(base64_image: str, new_size: list,
                             old_size: list) -> Tuple[str, str, str]:
    """Predict mask using a deep learning trained model
    Args:
        base64_image (str): base64 image encode
        new_size (list): original size of the image
        old_size (list): resize size of the new image
    Returns:
        Tuple[str, str]: return base64 masks for the original
        and the resized and the preview
    """
    # load image
    original_source_image = _base64_to_pil(base64_image)
    # resize to 224, 244
    resized_source_image = resize_with_pad(original_source_image,
                                           old_size)
    # tranform to tensor
    resized_source_tensor = _to_tensor(resized_source_image)
    # normalize
    preprocess_source_tensor = normalize_image(resized_source_tensor)
    # Predict mask
    mask = Predictor.predict(preprocess_source_tensor.unsqueeze(0)).sigmoid()
    # postprocess mask for resize image
    mask[mask > 0.8] = 1
    mask = mask.round().squeeze(0)
    # get original size mask image
    pil_resize_mask = _to_pil(mask)
    pil_original_mask = resize_with_pad(pil_resize_mask, new_size)
    # Combine model mask to grabcut algorithm
    # TODO
    # get preview image
    resized_preview_image = resized_source_tensor * mask
    resized_preview_image = _to_pil(resized_preview_image)
    # return bas64_image encode original mask, resize mask, and resize preview cut
    return (_pil_to_base64(pil_original_mask), 
            _pil_to_base64(pil_resize_mask),
            _pil_to_base64(resized_preview_image))

def get_bb_from_mask(mask: np.ndarray) -> np.ndarray:
    """Get bounding box xyxy format from image using mask, this will return
    the position where the important information from image are
    Args:
        mask (np.ndarray): numpy uint8 mask (H x W)
    Returns:
        np.ndarray: xyxy dimensions (x0, y0, x1, y1)
    """
    # get countours
    contours, _ = cv2.findContours(image=mask,
                                   mode=cv2.RETR_TREE,
                                   method=cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Get bounding box from contours
    c = contours[0]
    xywh_source = np.array(cv2.boundingRect(c), dtype=np.int32)
    return xywh2xyxy(xywh_source)

def get_blending_image(src: str, mask: str,
                       target: str, dims: list, naive: bool) -> Tuple[str, str]:
    """Apply blending algorithm to base 64 image encode
    Args:
        src (str): base 64 image encode source image
        mask (str): base 64 image encode mask image, this select the information
        target (str): base 64 image encode target image
        dims (list): xyxy list box dimensions where to paste the images
        naive (bool): specify if naive copy or deep image blending
    Returns:
        Tuple[str, str]: original blend image, blend image resize. Both in base64 encode
    """
    # Transform all the images into pil images
    # target images and dimensions are in 224 x 224 the resize version
    target = _to_tensor(_base64_to_pil(target))
    xyxy_target = np.array(dims, dtype=np.int32)
    # load mask and source in numpy
    source = _to_numpy(_base64_to_pil(src))
    mask = _to_numpy(_base64_to_pil(mask))
    # get xyxy dimensions for source using mask
    xyxy_source = get_bb_from_mask(mask)
    # extract mask and source information
    source = source[xyxy_source[1]:xyxy_source[3],
                    xyxy_source[0]:xyxy_source[2],:]
    mask = mask[xyxy_source[1]:xyxy_source[3],
                    xyxy_source[0]:xyxy_source[2]]
    # transfom to pil again
    source, mask = _to_pil(source), _to_pil(mask)
    h, w = get_dimensions_box(xyxy_target)
    # resize source and mask to fill h, w box dimensions
    source = resize_with_pad(source, new_shape=(h, w))
    mask = resize_with_pad(mask, new_shape=(h, w))
    # begin blend process
    blend_img = blend(source=_to_tensor(source), mask=_to_tensor(mask),
          target=target, dims=xyxy_target, naive=naive)
    # transform to pil 
    blend_img = _to_pil(blend_img)
    return 'Nada', _pil_to_base64(blend_img)