import json
import torch
import torchvision.transforms as T
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
from app.blending.models import MeanShift, VGG16_Model
import app.utils.general as G
from tqdm import tqdm

# std and mean configuration
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def pretty_print(obj):
    return json.dumps(obj, indent=3)

def normalize_image(tensor_image: torch.Tensor) -> torch.Tensor:
    normalize = T.Normalize(mean=mean,
                            std=std)
    return normalize(tensor_image)

def unormalize_image(tensor_image:torch.Tensor) -> torch.Tensor:
    mean = np.array(mean)
    std = np.array(std)
    unormalize = T.Normalize(mean=-mean/std, std=1/std)
    return unormalize(tensor_image)

def get_laplacian_kernel(device:torch.device = torch.device('cpu')) -> torch.Tensor:
    laplacian_kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=device)
    return laplacian_kernel.view(1, 1, *laplacian_kernel.shape)

def get_image_laplacian_operator(tensor_image: torch.Tensor, 
                                 device: torch.device = torch.device('cpu')) -> Tuple[torch.Tensor,
                                                            torch.Tensor,
                                                            torch.Tensor]:
    laplacian_kernel = get_laplacian_kernel(device)
    laplacian_conv = nn.Conv2d(in_channels=1,
                                out_channels=1,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
    laplacian_conv.weight = nn.Parameter(laplacian_kernel, requires_grad=False)
    rgb_channel_gradients = [laplacian_conv(tensor_image[:, ch, :, :].unsqueeze(1)) 
                             for ch in range(3)]
    return rgb_channel_gradients

def get_target_subimg(target: torch.Tensor,
                      mask: torch.Tensor,
                      dims: list):
    target_subimg = target[:, :, dims[0]:dims[1], dims[2]:dims[3]]
    return target_subimg

def get_mixing_gradients(image_data:dict,
                         device:torch.device = torch.device('cpu'),
                         alpha=0.5):
    source = image_data['source']
    mask = image_data['mask']
    dims = image_data['dims']
    target = image_data['target']
    # target = get_target_subimg(image_data['target'], mask, dims)
    
    rgb_source_gradients = [channel_gradient * mask 
                            for channel_gradient in get_image_laplacian_operator(source, device=device)]
    rgb_target_gradients = [channel_gradient * (1 - mask)
                            for channel_gradient in get_image_laplacian_operator(target, device=device)]
    # rgb_channel_target = [target[:, ch, :, :].unsqueeze(1) * (1 - mask) 
    #                       for ch in range(3)]
    # gradients_mix = [ source_channel * alpha + target_channel * (1 - alpha)
    #                  for source_channel, target_channel in zip(rgb_source_gradients,
    #                                                             rgb_channel_target)]
    gradients_mix = [ source_channel * alpha + target_channel * (1 - alpha)
                     for source_channel, target_channel in zip(rgb_source_gradients,
                                                                rgb_target_gradients)]
    return gradients_mix

def get_blending_gradients(tensor_image:torch.Tensor,
                           device:torch.device = torch.device('cpu')):
    # mask = image_data['mask']
    # dims = image_data['dims']
    # target = get_target_subimg(image_data['target'], mask, dims)
    
    # tensor_image_gradient = [channel_gradient * mask
    #                          for channel_gradient in get_image_laplacian_operator(tensor_image,
    #                                                                               device=device)]
    # rgb_channel_target = [target[:, ch, :, :].unsqueeze(1) * (1 - mask) 
    #                       for ch in range(3)]
    # gradient_blend = [ ch_img_gradient + ch_target_gradient
    #                   for ch_img_gradient, ch_target_gradient in zip(tensor_image_gradient,
    #                                                                  rgb_channel_target)]
    # gradient_blend = [channel_gradient 
    #                   for channel_gradient in get_image_laplacian_operator(tensor_image,
    #                                                                        device=device)]
    return get_image_laplacian_operator(tensor_image,
                                        device=device)

def execute_blend_process(source_temp: torch.Tensor, mask_temp: torch.Tensor,
                          target_temp: torch.Tensor, dims: List[int],
                          device: torch.device, naive: bool) -> torch.Tensor:
    """blend algorithm process
    Args:
        source_temp (torch.Tensor): source image tensor
        mask_temp (torch.Tensor): mask image tensor
        target_temp (torch.Tensor): target image tensor
        dims (List[int]): dimensions box for target in xyxy format 
        device (torch.device): torch device # TODO
        naive (bool): especify if just return the naive copy
    Returns:
        torch.Tensor: blend image
    """
    target = target_temp
    h, w = target.shape[2], target.shape[3]
    x0, y0, x1, y1 = dims
    
    source = torch.zeros_like(target)
    source[:, :, y0:y1, x0:x1] = source_temp
    
    mask = torch.zeros(1, 1, h, w)
    mask[:, :, y0:y1, x0:x1] = mask_temp
    
    input_img = torch.randn(*source.shape, device=device).contiguous()
    input_img.requires_grad = True
    
    # Pass all tensors to device
    target = target.to(device=device)
    mask = mask.to(device=device)
    source = source.to(device=device)
    
    # blend_img = input_img * mask + target * (1 - mask)
    naive_copy = source * mask + target * (1 - mask)
    
    new_image_data = {
        'mask': mask,
        'target': target,
        'source': source,
        'dims': dims
    }
    if naive:
        return naive_copy.squeeze(0)
    else:
        # Get ground truth gradients
        gt_gradients = torch.stack(get_mixing_gradients(new_image_data,
                                                        device=device), dim=2).squeeze(0)
        vgg16_features = VGG16_Model().to(device=device)
        mean_shift = MeanShift().to(device=device)
        
        # define optimizer and loss function
        optimizer = optim.LBFGS([input_img.requires_grad_()], lr=1.2, max_iter=200)
        mse_loss = nn.MSELoss().to(device=device)
        
        # Algorithms configuration
        run = [0]
        num_step = 500
        w_grad, w_cont, w_tv, w_style = 3e4, 3e1, 1e-6, 0.05
        configurations = {
            'num_step': num_step,
            'alg config': {
            'w_grad': w_grad,
            'w_cont': w_cont,
            'w_tv': w_tv,
            'w_style': w_style
            }
        }
        print(f'Blending algorithms configurations: {pretty_print(configurations)}')
        pbar = tqdm(total=num_step, desc='Blending operation ...', position=0)
        style_layers = vgg16_features.style_layers
        content_layers = vgg16_features.content_layers
        
        while run[0] < num_step:
            def closure():
                # zero grad optimizer
                optimizer.zero_grad()
                blend_img = (input_img * mask + target * (1 - mask))
                # gradient loss
                blend_gradients = torch.stack(get_blending_gradients(blend_img,
                                                                       device=device),
                                              dim=2).squeeze(0)
                loss_grad = w_grad * mse_loss(blend_gradients, gt_gradients) 
                # Content source loss
                input_features = vgg16_features(normalize_image(blend_img))
                source_features = vgg16_features(normalize_image(source))
                loss_content = 0
                for content_layer in content_layers:
                    loss_content += mse_loss(input_features[content_layer],
                                             source_features[content_layer])
                loss_content /= (len(content_layers)/w_cont)
                # Style source loss
                loss_source_style = 0
                for style_layer in style_layers:
                    loss_source_style += mse_loss(input_features[style_layer],
                                                  source_features[style_layer])
                loss_source_style /= (len(style_layers)/w_style)
                # TV Reg Loss
                loss_tv = w_tv * (torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + 
                            torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :])))
                # colect total loss
                loss_total = loss_grad + loss_content + loss_tv + loss_source_style
                if (run[0] + 1)%5 == 0 or (run[0] + 1 == 1):
                    with torch.no_grad():
                        # Save blend image in each iteration
                        # Cache
                        G.set_blend_img_in_cache(blend_img.squeeze(0))
                # Backward Optimization Step
                loss_total.backward()
                # Update pbar
                pbar_stats = {
                    "loss_grad": loss_grad.item(),
                    "loss_content": loss_content.item(),
                    "loss_source_style": loss_source_style.item(),
                    "loss_tv": loss_tv.item(),
                    "loss_total": loss_total.item()
                }
                pbar.set_postfix(**pbar_stats)
                pbar.update()
                # Update run
                run[0] += 1
                return loss_total
            optimizer.step(closure)
            with torch.no_grad():
                result_img = (input_img * mask + target * (1 - mask))
        return result_img.squeeze(0)

def blend(source: torch.Tensor, mask: torch.Tensor,
          target: torch.Tensor, dims: np.ndarray,
          naive: bool) -> torch.Tensor:
    """prepare source, mask, target and the target dimensions data
    and execute blend algorithm
    Args:
        source (torch.Tensor): source image tensor (Ch, H, W)
        mask (torch.Tensor): mask image tensor (Ch, H, W)
        target (torch.Tensor): target image tenros (Ch, H, W)
        dims (np.ndarray): xyxy np.ndarray box dimensions
        naive (bool): especify if naive copy or not
    Returns:
        torch.Tensor: blend image
    """
    mask = mask.unsqueeze(0)
    mask[mask > 0.4] = 1
    mask[mask <= 0.4] = 0
    source = source.unsqueeze(0)
    target = target.unsqueeze(0)
    return execute_blend_process(source_temp=source, mask_temp=mask,
                          target_temp=target, dims=[int(d) for d in dims],
                          device=torch.device('cpu'), naive=naive)




