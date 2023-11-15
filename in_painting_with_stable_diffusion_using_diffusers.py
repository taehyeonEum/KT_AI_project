# !pip install kaleido cohere openai tiktoken

# !pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install git+https://github.com/huggingface/diffusers.git

# from huggingface_hub import notebook_login

# notebook_login()
import argparse
import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import os
import cv2

import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

def inpainting(idx, result):

    device = "cuda"
    model_path = "runwayml/stable-diffusion-inpainting"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)

    import requests
    from io import BytesIO

    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows*cols

        w, h = imgs[0].size
        grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid

    # IMAGE_NAME = "img_f|table_sofa_cushion.jpg"
    # # IMAGE_NAME = "img_0|tv_cushion.jpeg"
    # # IMAGE_NAME = "img_1|drawer_bed_sofa.jpeg"
    # # IMAGE_NAME = "img_4|monitor_vase_keyboard_mouse.jpeg"
    # OBJECT_NAME = "Pillow_2" # img 0 
    # # OBJECT_NAME = "tv_0" # img 0 
    # # OBJECT_NAME = "bed_1" # img_1
    # # OBJECT_NAME = "drawer_2" # img_1
    # # OBJECT_NAME = "monitor_0" # img_4
    # bbox = [70, 305, 110, 335] # img_0 tv_0 bigger
    # bbox = [110 , 150, 511.93237, 511.6482] # img_1 bed_1 bigger -> fail
    # bbox = [ 25, 200, 150, 430 ] # img_1 drawer_2 bigger -> middle success
    # bbox = [ 175.44302, 150.84218, 370.87534, 370.24518] # img_4 monitor_2 bigger -> success

    IMAGE_NAME = result["i_name"]
    OBJECT_NAME = result["target"]
    BBOX = result["nbox"]


    IMAGE_PATH = os.path.join("./content", IMAGE_NAME)
    SOURCE_PATH = os.path.join("./outputs_grounded_sam", IMAGE_NAME, OBJECT_NAME)

    OUTPUT_BASE_DIR = f"./outputs_inpaintings"

    OUTPUT_DIR=os.path.join(OUTPUT_BASE_DIR, str(idx),  IMAGE_NAME, OBJECT_NAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from PIL import Image

    image = Image.open(IMAGE_PATH)
    image = image.resize((512, 512))
    image.save(os.path.join(OUTPUT_DIR, "original_image.jpg"))


    bbox = [int(x) for x in BBOX]
    x1, y1, x2, y2 = bbox

    #배경에 삽입될 OBJECT
    new_image= Image.open(os.path.join(SOURCE_PATH, "cropped_image.jpg"))
    # 바운딩 박스 크기 계산
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    # 바운딩 박스 크기로 새 이미지 조정
    new_image = new_image.resize((bbox_width, bbox_height))
    pasted_image = image
    pasted_image.paste(new_image, bbox)
    pasted_image.save(os.path.join(OUTPUT_DIR, "pasted_image.jpg"))

    templete = np.zeros((512, 512))
    contour_mask = Image.fromarray(templete)

    cropped_contour_mask = cv2.imread(os.path.join(SOURCE_PATH, "cropped_contour_mask.jpg"))
    pil_cropped_contour_mask = Image.fromarray(cropped_contour_mask)
    resized_ct_mask = pil_cropped_contour_mask.resize((bbox_width, bbox_height))

    contour_mask.paste(resized_ct_mask, bbox)

    print("contour_mask")
    print(contour_mask.size)

    # contour_mask.save(os.path.join(OUTPUT_DIR, "resized_cropped_contour_mask.jpg"), mode="RGB")
    # 
    # mask_image = Image.open("/content/output_image (2)_mask.png").resize((512, 512))

    prompt = "background"

    guidance_scale=9
    num_samples = 3
    generator = torch.Generator(device="cuda").manual_seed(1) # change the seed to get different results

    images = pipe(
        prompt=prompt,
        image=pasted_image,
        mask_image=contour_mask,
        guidance_scale=guidance_scale,
        generator=generator,
        num_images_per_prompt=num_samples,
    ).images

    for i, img in enumerate(images):
        # img = np.array(img)
        # print("type(img)")
        # print(type(img))
        # print("img.shape")
        # print(img.shape)
        # print("img.max() img.min()")
        # print(img.max(), img.min())
        # cv2.imwrite(os.path.join(OUTPUT_DIR, f"diff_result_{i}.jpg"), img)
        img.save(os.path.join(OUTPUT_DIR, f"diff_result_{i}.jpg"))



# insert initial image in the list so we can compare side by side

# images.insert(0, image)

# image_grid(images, 1, num_samples + 1)

# image_grid(images[:2],1,2)

# resized_images = [image.resize((850, 425)) for image in images]

# image_grid(resized_images[:2],1,2)

# image = Image.open("/content/image_sample.png")
# image

# # paste
# bbox = [380, 170, 520, 255]
# bbox = [int(x) for x in bbox]
# x1, y1, x2, y2 = bbox
# new_image= Image.open("/content/pillow.png")
# # 바운딩 박스 크기 계산
# bbox_width = x2 - x1
# bbox_height = y2 - y1
# # 바운딩 박스 크기로 새 이미지 조정
# new_image = new_image.resize((bbox_width, bbox_height))
# image.paste(new_image, bbox)
# image.save("output_image.jpg")

# image = image.resize((512, 512))
# image

# mask_image = Image.open("/content/output_image (1)_mask.png").resize((512, 512))
# mask_image

# prompt = "background"

# guidance_scale=9
# num_samples = 3
# generator = torch.Generator(device="cuda").manual_seed(1) # change the seed to get different results

# images = pipe(
#     prompt=prompt,
#     image=image,
#     mask_image=mask_image,
#     guidance_scale=guidance_scale,
#     generator=generator,
#     num_images_per_prompt=num_samples,
# ).images

# # insert initial image in the list so we can compare side by side
# images.insert(0, image)

# image_grid(images, 1, num_samples + 1)


