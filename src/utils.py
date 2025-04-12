import torch
import random
import importlib
import math
import PIL
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from typing import List


def cal_elasped_time(cur_time, start_time):
    elapsed_time = (cur_time - start_time)
    min, sec = divmod(elapsed_time, 60)
    if min > 60:
        hour, min = divmod(min, 60)
        time_ = f'{hour} h, {round(min, 0)} min, {round(sec, 0)} s'
    else:
        time_ = f'{round(min, 0)} min, {round(sec, 0)} s'
    return time_


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_obj_from_str(name: str, reload: bool = False):
    module, cls = name.rsplit(".", 1)

    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
        
    return getattr(importlib.import_module(module, package=None), cls)


def initialize_from_config(config: OmegaConf) -> object:
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def count_parameters(model: torch.nn.Module):
    full_network = sum([p.numel() for p in model.parameters() if p.requires_grad is True])
    params = {'Full network' : full_network}
    for module_name, module in model.named_children():
        params[module_name] = sum([p.numel() for p in module.parameters() if p.requires_grad is True])
    return params


def make_dirs(dir_paths):
    if not Path(dir_paths).exists():
        Path(dir_paths).mkdir(parents=True)
        
        
def make_image_grid(images: List[PIL.Image.Image], rows: int, cols: int, maxsize: int = None) -> PIL.Image.Image:
    """
    Clone from huggingface diffusers with modification
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if maxsize is not None:
        ori_w, ori_h = images[0].size
        if ori_h > ori_w:
            h = min(ori_h, maxsize)
            w = math.ceil(ori_w*(h/ori_h))
        else:
            w = min(ori_w, maxsize)
            h = math.ceil(ori_h*(w/ori_w))
            
        images = [img.resize((w, h)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


'''Taken from HiGAN/networks/utils.py'''
def pil_text_img(im, text, pos, color=(255, 0, 0), textSize=25):
    img_PIL = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    font = ImageFont.truetype('src/data/font/arial.ttf', textSize)
    fillColor = color  # (0,0,0)
    position = pos  # (100,100)
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, text, font=font, fill=fillColor)

    img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
    return img


def words_to_images(texts, img_h, img_w, n_channel=1):
    word_imgs = np.ones((len(texts), img_h, img_w, n_channel)).astype(np.uint8)
    for i in range(len(texts)):
        word_imgs[i] = pil_text_img(word_imgs[i], texts[i], (1, 1),  textSize=25)
    if n_channel > 1:
        word_imgs = word_imgs.astype(np.uint8)
    else:
        word_imgs = word_imgs.sum(axis=-1, keepdims=True).astype(np.uint8)
    word_imgs = torch.from_numpy(word_imgs).permute([0, 3, 1, 2]).float() / 128 - 1
    return word_imgs
