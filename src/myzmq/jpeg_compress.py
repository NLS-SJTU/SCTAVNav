#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import io

from PIL import Image
import numpy as np

def uint8_to_bytesio(ary):
    return io.BytesIO(ary)


def bytesio_to_uint8(bin_io):
    return np.fromstring(bin_io.getvalue(), dtype=np.uint8)


def uint8_to_pil(ary):
    return Image.fromarray(ary)


def pil_to_uint8(img_pil):
    return np.asarray(img_pil, dtype=np.uint8)


def img_rgb_to_jpeg(img_ary, quality=80):
    if(type(img_ary) == type(None)):
        return None
    if(type(img_ary) == type(np.array([1]))):
        img_pil = Image.fromarray(img_ary)
    else:
        img_pil = img_ary
    
    img_io  = io.BytesIO()              

    img_pil.save(img_io, format="JPEG", quality=quality)
    img_io.seek(0)
    bin_jpg=np.fromstring(img_io.getvalue(), dtype=np.uint8)
    return bin_jpg


def jpg_to_img_rgb(bin_jpg):
    bin_io = uint8_to_bytesio(bin_jpg)
    
    img_pil = Image.open(bin_io).convert('RGB')
    img_rgb = np.array(img_pil, dtype=np.uint8)
    return img_rgb


if __name__ == '__main__':
    
    sys.path.append('./')
    from global_cfg     import *
    from pygame_viewer  import *

    viewer=pygame_viewer_c()
    
    img_pil = Image.open("test.png")
    if not img_pil.mode == 'RGB': img_pil = img_pil.convert('RGB')
    
    img_uint8=pil_to_uint8(img_pil)
    #print('img_uint8.shape:',end='')
    print(img_uint8.shape)
    
    jpg_uint8=img_rgb_to_jpeg(img_uint8)
    #print('jpg_uint8.shape:',end='')
    print(jpg_uint8.shape)
    
    jpg_uint8.tofile('jpg_uint8.jpg')
    
    img_new=jpg_to_img_rgb(jpg_uint8)
    
    #viewer.show_img_u8(cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB))
    viewer.show_img_u8(img_new)
    while True:
        if not viewer.poll_event():
            break
    exit()
