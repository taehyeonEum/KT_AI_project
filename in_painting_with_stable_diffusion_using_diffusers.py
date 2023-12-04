# !pip install kaleido cohere openai tiktoken

# !pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install git+https://github.com/huggingface/diffusers.git

# from huggingface_hub import notebook_login

# notebook_login()
from PIL import Image
import argparse
import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import os
import cv2
import math

import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from InpaintAnything import remove_anything


def make_rectangle_d_mask(bbox):
    # 이미지 크기 설정
    image_size = (512, 512)

    # 넘파이 배열 생성 (전체를 0으로 초기화)
    image = np.zeros(image_size)
    image_b = np.zeros(image_size)

    # bounding box 좌표
    x_min, y_min, x_max, y_max = bbox

    # 좌측상단 올림 우측 하단 내림.
    x_min = math.floor(x_min)
    y_min = math.floor(y_min)
    x_max = math.ceil(x_max)
    y_max = math.ceil(y_max)

    image[y_min:y_max, x_min:x_max] = 256

    
    print("_________orginal squeare: x_min y_min x_max y_max________")
    print(x_min, y_min, x_max, y_max)

    # make mask thicker
    if x_min > 15: 
        x_min = x_min - 15
    # else:
    #     x_min = x_min - 10

    if y_min > 15: 
        y_min = y_min - 15
    # else: 
    #     y_min = y_min - 10

    if x_max < (512-15): 
        x_max = x_max + 15
    # else: 
    #     x_max = x_max + 10

    if y_max < (512-15): 
        y_max = y_max + 15
    # else:
    #     y_max = y_max + 10


    print("_________bigger squeare: x_min y_min x_max y_max________")
    print(x_min, y_min, x_max, y_max)

    # bounding box 안의 영역을 256으로 설정
    image_b[y_min:y_max, x_min:x_max] = 256
    image_d = image_b-image
    image_d = image_d[y_min:y_max, x_min:x_max]
    image_d = Image.fromarray(image_d)

    return image_d, [x_min, y_min, x_max, y_max]

def inpainting(idx, result, ex_num):


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

    EX_ID = f"EX_{ex_num}"
    IMAGE_NAME = result["i_name"]
    OBJECT_NAME = result["target"]
    BBOX = result["nbox"]
    ORIGINAL_BBOX = result["original_bbox"]

    IMAGE_PATH = os.path.join("./content", IMAGE_NAME)
    SOURCE_PATH = os.path.join("./outputs_grounded_sam_suppliments", EX_ID, IMAGE_NAME, OBJECT_NAME)

    OUTPUT_BASE_DIR = f"./outputs_inpaintings_suppliments"

    OUTPUT_DIR=os.path.join(OUTPUT_BASE_DIR, EX_ID, IMAGE_NAME, OBJECT_NAME, str(idx))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    OUTPUT_MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


    image = Image.open(IMAGE_PATH)
    image = image.resize((512, 512))
    IMAGE_PATH_512 = os.path.join(OUTPUT_DIR, "original_image.jpg")
    image.save(IMAGE_PATH_512)

    x_min, y_min, x_max, y_max = ORIGINAL_BBOX
    mean_x = (x_min+x_max) / 2
    mean_y = (y_min+y_max) / 2
    
    remove_anything.remove_anything(IMAGE_PATH_512 , "key_in" , [mean_x, mean_y] , [1], 15 , OUTPUT_DIR 
                    , "vit_h" 
                    , "./checkpoints/sam_vit_h_4b8939.pth" 
                    , "./lama/configs/prediction/default.yaml"
                    , "./InpaintAnything/pretrained_models/big-lama", result )

    # code for remove object with origianl code..!! -> replaced by inpaint anything. 
    # # prompt = "wall"
    # # prompt = f"remove {OBJECT_NAME.split('_')[0]} from background"
    # # prompt = f"Remove {OBJECT_NAME.split('_')[0]} from the background and keep it clean without leaving any other objects in the background."
    # # prompt = ""
    # # prompt = "room with nothing"
    # # prompt = "background with nothing else"
    # prompt = "Remove background and objects naturally" # EX_10
    # prompt = "Remove objects naturally" # EX_11
    # prompt = "Remove objects naturally and fill blank naturally" # EX_12
    # print('\n\n\n*******remove object inpainting prompt*******')
    # print(prompt)
    # guidance_scale=9
    # num_samples = 1
    # generator = torch.Generator(device="cuda").manual_seed(23) # change the seed to get different results
    
    # # # 딱 맞는 object mask적용
    # # object_mask = Image.open(os.path.join(SOURCE_PATH, "mask.jpg"))
    # # object_mask = np.array(Image.open(os.path.join(SOURCE_PATH, "mask.jpg")))
    # # #object_mask array 로 변환하고 입력받기.
    # # object_mask = np.where(object_mask==256, 1, object_mask)
    # # #object_mask 다시 PIL타입으로 저장.
    # # object_mask = Image.fromarray(object_mask)

    # # 조금 큰 object mask 적용
    # object_mask = Image.fromarray(create_mask_with_bounding_box(ORIGINAL_BBOX))
    # object_mask = object_mask.convert("L")
    # object_mask.save(os.path.join(OUTPUT_DIR, "remove_object_square_mask.jpeg"))
    # # object_mask = np.where(object_mask==256, 1, object_mask)

    # object_removed_images = pipe(
    # prompt=prompt,
    # image=image,
    # mask_image=object_mask,
    # guidance_scale=guidance_scale,
    # generator=generator,
    # num_images_per_prompt=num_samples,).images
    
    # object_removed_image = object_removed_images[0]
    # object_removed_image.save(os.path.join(OUTPUT_DIR, "object_removed_image.jpg"))
    # image = object_removed_image


    ia_images = []
    image_paths = []
    for i in range(3):
        path = f"{OUTPUT_DIR}/original_image/inpainted_with_mask_{i}.png"
        image_paths.append(path)
        ia_images.append(Image.open(path))


    bbox = [int(x) for x in BBOX]
    x1, y1, x2, y2 = bbox

    #배경에 삽입될 OBJECT
    new_image= Image.open(os.path.join(SOURCE_PATH, "cropped_image.jpg"))
    # 바운딩 박스 크기 계산
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    # 바운딩 박스 크기로 새 이미지 조정
    new_image = new_image.resize((bbox_width, bbox_height))

    for bi, image in enumerate(ia_images):
        pasted_image = image
        pasted_image.paste(new_image, bbox)
        pasted_image.save(os.path.join(OUTPUT_DIR, "pasted_image.jpg"))

        templete = np.zeros((512, 512))
        contour_mask = Image.fromarray(templete)

        cropped_contour_mask = cv2.imread(os.path.join(SOURCE_PATH, "cropped_contour_mask.jpg"))
        cropped_contour_mask = np.where(cropped_contour_mask==256, 1, cropped_contour_mask)
        pil_cropped_contour_mask = Image.fromarray(cropped_contour_mask)
        resized_ct_mask = pil_cropped_contour_mask.resize((bbox_width, bbox_height))

        #얇은 사각형 도넛 모양의 마스크를 생성함 , 저장함. 
        mask_d, nbbox = make_rectangle_d_mask(BBOX)
        mask_d=mask_d.convert("L")
        mask_d.save(os.path.join(OUTPUT_MASK_DIR, "mask_D.jpeg"))

        #반전 마스크를 구성함.
        contour_mask.paste(resized_ct_mask, bbox)
        contour_mask = contour_mask.convert("L")
        contour_mask.save(os.path.join(OUTPUT_MASK_DIR, "contour_mask.jpeg"))

        #반전 마스크에 얇은 사각형 도넛 모양의 마스크를 붙임.
        contour_mask.paste(mask_d, nbbox)
        contour_mask.save(os.path.join(OUTPUT_MASK_DIR, "thick_contour_mask.jpeg"))

        # 최종 반전 마스크를 구성함.
        contour_mask.paste(resized_ct_mask, bbox)
        contour_mask.save(os.path.join(OUTPUT_MASK_DIR, "final_contour_mask.jpeg"))



        # print("contour_mask")
        # print(contour_mask.size)

        prompt = "background"

        guidance_scale=9
        num_samples = 1
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
            img.save(os.path.join(OUTPUT_DIR, f"diff_result_{bi}_{i}.jpg"))
    return OUTPUT_DIR


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


