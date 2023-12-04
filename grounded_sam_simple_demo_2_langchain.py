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
import re

import torch
import torchvision

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


def gpt_grounded_sam(image_nname, ex_num, question, input_dir, output_dir, openai_api):

    template = """
    Extract only objects names. 
    Do not include 'answer,' just extract the object names.

    Question: {question} 
    """
    OPEN_AI_API_KEY =  openai_api

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = OpenAI(model_name="gpt-3.5-turbo",openai_api_key=OPEN_AI_API_KEY)
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # question = "Please reduce the area of the pillow located on the far right and move it to the left side of the sofa."
    # question = "Please reduce the area of the pillow located at the far right and move it onto the table in front of the sofa."
    # question = "Make the tv much bigger."

    print("\n\n\n////////////////////////////////////")
    print("llm_chain_run(question)")
    output_string= llm_chain.run(question)
    print("-------raw GPT output----------")
    print(output_string)

    # '\nAnswer:' 부분 제거
    # output_string = output.split(': ')[1] # 해당 값 grounding dino로 보내기 

    # 결과 출력
    
    lang_classes = output_string.split(", ")
    for i, clas in enumerate(lang_classes):
        lang_classes[i] = clas.replace(" ", "").replace("\n","").lower()
    print("------regularized classes names------")
    print(lang_classes)

    IMAGE_NAME = image_nname
    print("IMAGE_NAME")
    print(IMAGE_NAME)
    
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

    IMAGE_DIR = input_dir

    # Predict classes and hyper-param for GroundingDINO

    # get image info with current image number
    # SOURCE_IMAGE_PATH = "./assets/demo2.jpg"
    # image_names = os.listdir(IMAGE_DIR)
    # image_numbers = []
    # image_paths = []
    # image_classes = []

    # for name in image_names:
    #     image_numbers.append(name.split("|")[0])
    #     objs=((name.split("|")[1]).split(".")[0]).split("_")
    #     image_classes.append(objs)
    #     image_paths.append(os.path.join(IMAGE_DIR,name))

    # print(image_names)
    # print(image_numbers)
    # print(image_paths)
    # print(image_classes)
    # SOURCE_IMAGE_PATH = image_paths[CURRENT_IMG_NUM]

    OUTPUT_DIR = f"{output_dir}/EX_{ex_num}/{IMAGE_NAME}"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SOURCE_IMAGE_PATH = os.path.join(IMAGE_DIR, IMAGE_NAME)
    # CLASSES = image_classes[CURRENT_IMG_NUM]
    CLASSES = lang_classes
    BOX_THRESHOLD = 0.35
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
    # print("shape after resize and np.array")
    # print(type(image)) # <class 'numpy.ndarray'>
    # print(image.shape) # (512, 512, 3)


    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    print("\n\n\n////////////////////////////////////")
    print("GroundingDINO outputs")
    print(detections) # detections = xyxy, mask, confidence, class_id, tracker_id
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
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"groundingdino_annotated_image_{IMAGE_NAME}.jpg"), cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

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
    # masks = np.squeeze(masks)
    # print(mask.max()) # 1
    masks = masks * 256

    # mask  자체가 ndarray의 형태로 출력됨. 
    # print("////////////////////////////////////")
    # print("masks")
    # # print(masks)
    # print(type(masks)) # <class 'numpy.ndarray'>
    # print(masks.shape) # (3, 512, 512)
    # print("/////////////////////////////////////\n\n\n")

    #detections.mask 에 다시 변형된 mask를 저장
    # detections.mask = masks



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
        
        # make mask thicker
        # if x_min > 30: 
        #     x_min = x_min - 30
        # if y_min > 30: 
        #     y_min = y_min - 30
        # if x_max < (512-30): 
        #     x_max = x_max + 30
        # if y_max < (512-30): 
        #     y_max = y_max + 30

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
        # make mask thicker
        # if x_min > 30: 
        #     x_min = x_min - 30
        # if y_min > 30: 
        #     y_min = y_min - 30
        # if x_max < (512-30): 
        #     x_max = x_max + 30
        # if y_max < (512-30): 
        #     y_max = y_max + 30
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        return cropped_image, cropped_mask

    cls_bbox = {}
    square_masks = []
    anti_masks = []
    contour_masks = []
    cropped_images = []
    cropped_contour_masks=[]
    lang_chain_input = []

    for i in range(len(detections.xyxy)):

        # print("mask.max()")
        # print(mask.max()) # 256
        # print(mask.min()) # 0

        mask = masks[i]
        class_name = CLASSES[detections.class_id[i]].lower()
        bbox = detections.xyxy[i]

        cls_bbox[f'{class_name}_{i}'] = bbox

        x_min, y_min, x_max, y_max = bbox
        line = f"{class_name}_{i}" + " [" + str(x_min) + " " + str(y_min) + " " + str(x_max) + " " + str(y_max) + ']'
        lang_chain_input.append(line)

        os.makedirs(os.path.join(OUTPUT_DIR, f"{class_name}_{i}"), exist_ok=True)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "mask.jpg"), mask)

        square_mask = create_mask_with_bounding_box(bbox)
        square_masks.append(square_mask)

        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "square_mask.jpg"), square_mask)

        anti_mask = create_antiMask_with_originalMask(mask)
        anti_masks.append(anti_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "anti_mask.jpg"), anti_mask)
        
        contour_mask = (square_mask + anti_mask) - 256
        contour_masks.append(contour_masks)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "contour_mask.jpg"), contour_mask)

        cropped_image, cropped_contour_mask = save_cropped_image_mask(image, contour_mask, bbox)    
        cropped_images.append(cropped_image)
        cropped_contour_masks.append(cropped_contour_mask)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "cropped_image.jpg"), cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{class_name}_{i}", "cropped_contour_mask.jpg"), cropped_contour_mask)


        # print("mask.shape, anti_mask.shape, contour_mask.shape")
        # print(mask.shape, anti_mask.shape, contour_mask.shape) # (512, 512) (512, 512) (512, 512)
        # print("mask.sum(), anti_mask.sum(), contour_mask.sum()")
        # print(mask.sum(), anti_mask.sum(), contour_mask.sum()) # 610048 66498816 119552.0
        # print("type(mask), type(anti_mask), type(contour_mask)")
        # print(type(mask), type(anti_mask), type(contour_mask)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'> <class 'numpy.ndarray'>
        # print(mask)
        # print(anti_mask)
        # print(contour_mask)
        

    print("--------lang_chain_input-------")
    print(lang_chain_input)

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
    OUTPUT_DIR = f"./outputs_grounded_sam/{IMAGE_NAME}"
    cv2.imwrite(os.path.join(OUTPUT_DIR,f"grounded_sam_annotated_image_{IMAGE_NAME}.jpg"), cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    class CommaSeparatedListOutputParser(BaseOutputParser):
        def parse(self, text: str):
            return text.strip().split(", ")

    old_template = """
    Let’s imagine you’re a semantic parser. I’ll provide specific bounding box coordinates and
    instructions. These coordinates represent the [(left corner x, y values), (right corner x, y values)]
    of the bounding box. Extract the action and target from the instructions and new bounding
    box coordinates, noting the association between bounding box coordinates and the target. Also,
    bounding box coordinates should not exceed the frame size [(0,0), (512, 512)].  Then, adjust the
    bounding box using expected values and display the resulting coordinates as the output. Let’s think step by step.

    condition : 
    First, please maintain the width-to-height ratio of the box.
    Second, unless the object is moving, please execute the insturction with the left corner x, y values of the box fixed.
    Third, represent the coordinates as integers.

    For example:
    
    insturction = The pillow located on the far right is reduced in area and moved to the left of the sofa

    Bounding Box Coordinates: [
        'Sofa_0 [105 226 363 387]', 
        'Table_1 [45 284 102 378]', 
        'Pillow_2 [257 227 297 300]', 
        'Pillow_3 [139 224 172 297]', 
        'Pillow_4 [170 226 208 301]', 
        'Table_5 [174 348 294 439] 
        ]
    
    output = [
        'action': 'the pillow located on the far right is reduced in area','target': 'pillow_2',  nbox': [(427 188 467 218)],
        'action': 'moved to the left of the sofa','target': 'pillow_2', 'nbox': [(174 185 214 215)]
    ]"""

    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # human_template = """
    # Instruction : {instruction}. Please remind the condition.

    # Bounding Box Coordinates: {bounding_box_coordinates}

    # The output is returned in the form of a dictionary.
    # Output:"""

    #few shot template!
    template = """
    Let’s imagine you’re a semantic parser. I’ll provide specific bounding box coordinates and
    instructions. These coordinates represent the [(left corner x, y values), (right corner x, y values)]
    of the bounding box. Extract the action and target from the instructions and new bounding
    box coordinates, noting the association between bounding box coordinates and the target. Also,
    bounding box coordinates should not exceed the frame size [(0,0), (512, 512)].  Then, adjust the
    bounding box using expected values and display the resulting coordinates as the output. Let’s think step by step.

    condition : 
    First, please maintain the width-to-height ratio of the box.
    ecute the instruction with the left corner x, y values of the box fixed unless the object is moving.
    Third, represent the coordinates as integers.

    For example:
    
    Bounding Box Coordinates: [
        'Sofa_0 [105 226 363 387]', 
        'Table_1 [45 284 102 378]', 
        'Pillow_2 [257 227 297 300]', 
        'Pillow_3 [139 224 172 297]', 
        'Pillow_4 [170 226 208 301]', 
        'Table_5 [174 348 294 439] 
    ]
        
    Example 1:
    Instruction: "Enlarge the rightmost table."
    Original Bounding Box: 'Table_5 [174 348 294 439]'
    Output: ['action': 'enlarged', 'target': 'Table_5', 'original_box': [(174, 348, 294, 439)], 'new_box': [(164, 338, 304, 449)]]

    Example 2:
    Instruction: "Move the leftmost pillow down."
    Original Bounding Box: 'Pillow_3 [139 224 172 297]'
    Output:  ['action': 'moved down', 'target': 'Pillow_3', 'original_box': [(139, 224, 172, 297)], 'new_box': [(139, 214, 172, 287)]]
    Example 3:
    Instruction: "Shift the middle pillow to the right."
    Original Bounding Box: 'Pillow_4 [170 226 208 301]'
    Output: ['action': 'moved to the right', 'target': 'Pillow_4', 'original_box': [(170, 226, 208, 301)], 'new_box': [(190, 226, 228, 301)]]

    Example 4:
    Instruction: "Reduce the size of the leftmost table."
    Original Bounding Box: 'Table_1 [45 284 102 378]'
    Output: ['action': 'reduced in size', 'target': 'Table_1', 'original_box': [(45, 284, 102, 378)], 'new_box': [(55, 294, 92, 368)]]
        
    """

    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = """
    Instruction : {instruction}. Please remind the condition.

    Bounding Box Coordinates: {bounding_box_coordinates}

    The output is returned in the form of a dictionary.
    The bounding box coordinates should be interpreted as [(bottom left x, y), (top right x, y)]
    
    Output:
    
    """


    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(openai_api_key = OPEN_AI_API_KEY, temperature=1) | CommaSeparatedListOutputParser()
    output= chain.invoke({"instruction": question, "bounding_box_coordinates": ','.join(lang_chain_input)})

    print("\n\n\n---------gpt_raw_output---------")
    print(output)

    formatted_text_string = " ".join(output)

    print(formatted_text_string)
    print(type(formatted_text_string))

    input_string = formatted_text_string


    # 정규 표현식을 사용하여 target 값과 nbox 값 추출
    targets = re.findall(r"'target': '(.*?)'", input_string)
    for i, target in enumerate(targets): 
        targets[i] = str(target).lower()
    print("targets")
    print(targets)
    nbox_values = re.findall(r"\(([\d.]+ [\d.]+ [\d.]+ [\d.]+)\)", input_string)

    # nbox 값을 실수로 변환하고 리스트로 구성
    nbox_list = [[float(coord) for coord in box.split()] for box in nbox_values]
    print(nbox_list)
    # 결과 딕셔너리 생성
    # result = [{'target': targets[0], 'nbox': nbox_list[0]},
    #         {'target': targets[1], 'nbox': nbox_list[1]}]
    
    print("\n\n\n*************info_dict**************")
    print(cls_bbox)

    result = []
    print("len(targets)")
    print(len(targets))
    for i in range(len(targets)):
        result.append({
            'i_name':IMAGE_NAME, 
            'target': targets[i],
            'question': question, 
            'nbox': nbox_list[i], 
            "original_bbox": cls_bbox[targets[i]] 
            })

    # 결과 출력
    print("\n\n\n----------final_result-----------")
    print(result)

    f=open("./txt_s/results_2.txt", 'a')
    f.write(str(ex_num)+"\n")
    for r in result:
        f.write(f"   {str(r)}\n")
    f.close()

    return result


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_name", type=str)

    args = parser.parse_args()


    gpt_grounded_sam(args.image_name)