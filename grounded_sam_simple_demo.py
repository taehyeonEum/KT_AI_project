import argparse
import datetime
now = datetime.datetime.now()
now = str(now)
print(now)

import os 
import cv2
import numpy as np
import math
import supervision as sv
from PIL import Image

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

def main(current_img_num):
    CURRENT_IMG_NUM = current_img_num
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./checkpoints/sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)

    IMAGE_DIR = "./content"

    # Predict classes and hyper-param for GroundingDINO
    # SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
    image_names = os.listdir(IMAGE_DIR)
    image_numbers = []
    image_paths = []
    image_classes = []
    for name in image_names:
        image_numbers.append(name.split("|")[0])
        objs=((name.split("|")[1]).split(".")[0]).split("_")
        image_classes.append(objs)
        image_paths.append(os.path.join(IMAGE_DIR,name))

    print(image_names)
    print(image_numbers)
    print(image_paths)
    print(image_classes)

    OUTPUT_DIR = f"./outputs_image_name/{image_names[CURRENT_IMG_NUM]}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SOURCE_IMAGE_PATH = image_paths[CURRENT_IMG_NUM]
    CLASSES = image_classes[CURRENT_IMG_NUM]
    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.4
    NMS_THRESHOLD = 0.8


    # load image
    # image = cv2.imread(SOURCE_IMAGE_PATH)
    # load image with PIL
    image = Image.open(SOURCE_IMAGE_PATH)

    #diffusion을 위해 512x512로 이미지 크기 수정
    # print("type(PIL image)")
    # print(type(image)) # <class 'PIL.JpegImagePlugin.JpegImageFile'>
    image = np.array(image.resize((512, 512)))
    # print("image.shape after resize and np.array")
    # print(type(image)) # <class 'numpy.ndarray'>
    # print(image.shape) # (512, 512, 3)


    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    print("////////////////////////////////////")
    print("detections")
    print(detections) # detections = xyxy, mask, confidence, class_id, tracker_id
    print("////////////////////////////////////\n\n\n")
    # Detections(xyxy=array([[204.80188 , 146.85254 , 298.17764 , 223.25366 ],ㅇ
    #        [388.03534 , 290.52283 , 432.3512  , 323.94464 ],
    #        [ 78.035995, 299.67517 , 127.26997 , 321.89618 ]], dtype=float32), mask=None, confidence=array([0.8248828 , 0.5592095 , 0.50679314], dtype=float32), class_id=array([0, 1, 1]), tracker_id=None)


    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # save the annotated(1:1로 지명된) grounding dino image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"groundingdino_annotated_image_{image_numbers[CURRENT_IMG_NUM]}.jpg"), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

    # print("####################################")
    # print("type(annotated_frame)")
    # print(type(annotated_frame)) # <class 'numpy.ndarray'>
    # print(annotated_frame.shape) # (512, 512, 3)
    # print("####################################\n\n\n")

    # NMS post process
    # print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # print("####################################") # 3
    # print(type(detections.xyxy)) # <class 'numpy.ndarray'>
    # print("type(detections.xyxy)") # type(detections.xyxy)
    # print(detections.xyxy)
    #                         # [[204.80188  146.85254  298.17764  223.25366 ]
    #                         #  [388.03534  290.52283  432.3512   323.94464 ]
    #                         #  [ 78.035995 299.67517  127.26997  321.89618 ]]
    # print("####################################\n\n\n")

    # print(f"After NMS: {len(detections.xyxy)} boxes") # 3

    # Prompting SAM with detected boxes
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )


    masks = np.array(detections.mask, dtype=np.uint8)
    masks = np.squeeze(masks)
    # print(mask.max()) # 1
    masks = masks * 256

    # mask  자체가 ndarray의 형태로 출력됨. 
    print("////////////////////////////////////")
    print("masks")
    # print(masks)
    print(type(masks)) # <class 'numpy.ndarray'>
    print(masks.shape) # (3, 512, 512)
    print("/////////////////////////////////////\n\n\n")

    #detections.mask 에 다시 변형된 mask를 저장
    detections.mask = masks


    square_masks = []
    anti_masks = []
    contour_masks = []
    cropped_images = []
    cropped_contour_masks=[]

    # bounding box 내부를 채운 mask 출력
    def create_mask_with_bounding_box(bbox):
        # 이미지 크기 설정
        image_size = (512, 512)

        # 넘파이 배열 생성 (전체를 0으로 초기화)
        image = np.zeros(image_size)

        # bounding box 좌표
        x_min, y_min, x_max, y_max = bbox

        # 좌측상단 올림 우측 하단 내림.
        x_min = math.floor(x_min)
        y_min = math.floor(y_min)
        x_max = math.ceil(x_max)
        y_max = math.ceil(y_max)

        # print("x_min y_min x_max y_max")
        # print(x_min, y_min, x_max, y_max)

        # bounding box 안의 영역을 256으로 설정
        image[y_min:y_max, x_min:x_max] = 256

        return image

    # object를 제외한 부분이 채워진 mask 출력
    def create_antiMask_with_originalMask(original_mask):
    
        original_mask = np.where(original_mask==0, 512, original_mask)
        original_mask = original_mask - 256
        return original_mask 

    # contour mask 출력
    def save_cropped_image_mask(image, mask, bbox):
        x_min, y_min, x_max, y_max = bbox

        # 좌측상단 올림 우측 하단 내림.
        x_min = math.floor(x_min)
        y_min = math.floor(y_min)
        x_max = math.ceil(x_max)
        y_max = math.ceil(y_max)

        cropped_image = image[y_min:y_max, x_min:x_max, :]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        return cropped_image, cropped_mask


    for i in range(len(detections.xyxy)):

        # print("mask.max()")
        # print(mask.max()) # 256
        # print(mask.min()) # 0

        mask = detections.mask[i]
        class_name = CLASSES[detections.class_id[i]]
        bbox = detections.xyxy[i]
        cv2.imwrite(os.path.join(OUTPUT_DIR,f"mask_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), mask)

        square_mask = create_mask_with_bounding_box(bbox)
        square_masks.append(square_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR,f"square_mask_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), square_mask)

        anti_mask = create_antiMask_with_originalMask(mask)
        anti_masks.append(anti_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR,f"anti_mask_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), anti_mask)
        
        contour_mask = (square_mask + anti_mask) - 256
        contour_masks.append(contour_masks)
        cv2.imwrite(os.path.join(OUTPUT_DIR,f"contour_mask_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), contour_mask)

        cropped_image, cropped_contour_mask = save_cropped_image_mask(image, contour_mask, bbox)    
        cropped_images.append(cropped_image)
        cropped_contour_masks.append(cropped_contour_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"cropped_image_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"cropped_contour_mask_{image_numbers[CURRENT_IMG_NUM]}_{class_name}_{i}.jpg"), cropped_contour_mask)


        # print("mask.shape, anti_mask.shape, contour_mask.shape")
        # print(mask.shape, anti_mask.shape, contour_mask.shape) # (512, 512) (512, 512) (512, 512)
        # print("mask.sum(), anti_mask.sum(), contour_mask.sum()")
        # print(mask.sum(), anti_mask.sum(), contour_mask.sum()) # 610048 66498816 119552.0
        # print("type(mask), type(anti_mask), type(contour_mask)")
        # print(type(mask), type(anti_mask), type(contour_mask)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(mask)
        # print(anti_mask)
        # print(contour_mask)
        


    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    cv2.imwrite(os.path.join(OUTPUT_DIR,f"grounded_sam_annotated_image_{image_numbers[CURRENT_IMG_NUM]}.jpg"), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)) 

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--curr_img_num", type=int)

    args = parser.parse_args()

    main(args.curr_img_num)