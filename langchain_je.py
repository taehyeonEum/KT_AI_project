import datetime
now = datetime.datetime.now()
now = str(now)
print(now)

import os
os.makedirs(f"./outputs/{now}")

# # (1) langchain 
# key : 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

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


import supervision as sv

from PIL import Image
import cv2

import re

import torch

import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

template = """
Extract only objects names.

Question: {question} 

"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm = OpenAI(model_name="text-davinci-003",openai_api_key="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")
llm_chain = LLMChain(prompt=prompt, llm=llm)

# question = "Please reduce the area of the pillow located on the far right and move it to the left side of the sofa."
question = "Please reduce the area of the pillow located at the far right and move it onto the table in front of the sofa."

print("////////////////////////////////////")
print("llm_chain_run(question)")
output= llm_chain.run(question)
print(output)
print("////////////////////////////////////")

# '\nAnswer:' 부분 제거
output_string = output.split('Answer: ')[1] # 해당 값 grounding dino로 보내기 

# 결과 출력
print(output_string)

# # (2) Grounding DINO 

# get_ipython().run_line_magic('cd', '/home/libra/vscode_ji/kt_project')
HOME = os.getcwd()
print(HOME)

# get_ipython().run_line_magic('cd', '{HOME}')
# !git clone https://github.com/IDEA-Research/GroundingDINO.git
# get_ipython().run_line_magic('cd', '{HOME}/GroundingDINO')
# !pip install -q -e .
# !pip install -q roboflow

CONFIG_PATH = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"

# print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
# get_ipython().run_line_magic('cd', '{HOME}')
# get_ipython().system('mkdir {HOME}/weights')
# get_ipython().run_line_magic('cd', '{HOME}/weights')
# !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


# ----

# (2) LLM에서 출력된 object name (변수명: output_string)을 DINO로 detection 진행

# In[ ]:


WEIGHTS_PATH = "./checkpoints/groundingdino_swint_ogc.pth"
# WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
# print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
# get_ipython().run_line_magic('cd', '{HOME}/GroundingDINO')

from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

IMAGE_NAME = "image_first|table_sofa_cushion.jpg"
IMAGE_PATH = "./content_512/" + IMAGE_NAME
# print(IMAGE_PATH)

# image = Image.open(IMAGE_PATH)
# image = np.array(image.resize((512, 512)))
# IMAGE_NAME = IMAGE_NAME.split('.')[0]+"_`.jpg"
# IMAGE_PATH = "./content/" + IMAGE_NAME
# cv2.imwrite(f"./content/{IMAGE_NAME}", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


# 나열된 단어에 해당하는 object에 bounding box값 출력

TEXT_PROMPT = output_string
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.1

print(TEXT_PROMPT)

image_source, image = load_image(IMAGE_PATH)
# image_source shape : (806, 808, 3)
# iamge_source type : np.adarray
# image shape : [3, 800, 801]
# image type : torch.Tensor
# image is normalized image..!!

print("##########################################")
print("image_source.shape")
print(image_source.shape) 
print(type(image_source))
IMG_LOG = "./image_log"
cv2.imwrite(os.path.join(IMG_LOG, "image_source.png"), image_source)
print('image.shape')
print(image.shape) # [3, 800, 801]
cv2.imwrite(os.path.join(IMG_LOG, "image.jpg"), (np.array(image).transpose(1, 2, 0) * 256))
print("##########################################")


boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD
)

print("\n\n\n************************************")
print("boxes, logits, phrases")
print(boxes, logits, phrases)
print("************************************\n\n\n")


# In[1258]:


# 해당 bbox값 도식화 

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# get_ipython().run_line_magic('matplotlib', 'inline')
# sv.plot_image(annotated_frame, (10, 10))
cv2.imwrite(f"./outputs/{now}/{IMAGE_NAME.split('.')[0]}_detection_output.jpg", annotated_frame)


# In[1355]:


# Image.fromarray(mask)


# ---

# (2) box coordinates 변환

# In[1348]:


import torch
from torchvision.ops import box_convert


# In[ ]:


print(image_source.shape) # (H, W, C)
source_h, source_w, _ = image_source.shape
print(source_h, source_w)   


# In[1351]:


boxes # 기존 bbox 값 -> normalized 되어 있어져 있어 수정 필요  


# In[1353]:


import torch
from torchvision.ops import box_convert

def convert_xy(boxes):
  boxes_xy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
  return boxes_xy

import numpy as np
h, w, _ = image_source.shape
boxes0 = boxes * torch.Tensor([w, h, w, h])
mask = np.zeros_like(image_source)

for box in convert_xy(boxes0):
  x0, y0, x1, y1 = box
  mask[int(y0):int(y1), int(x0):int(x1), :] = 255


# In[1354]:


# 변환된 bbox 값 도출 
convert_xy(boxes0)


# ---

# (3) 탐지된 객체의 이름을 구분하기 위해 객체이름 등장횟수 계산 및 이름 재정의 (objectname_number)

# In[ ]:


# 각 객체 이름의 등장 횟수를 계산

from collections import Counter

list_temp = []
for i in Counter(phrases).values() : 
    for j in range(1,i+1):
        list_temp.append(j)
print('결과', list_temp)


# In[1346]:


labels = [
        f"{phrase}_{count} {boxes}"
        for phrase, count, boxes
        in zip(phrases,list_temp, convert_xy(boxes0).numpy().astype(int))
    ]
print(phrases)
print(labels)
# [i for i in labels]
# print([i for i in labels].replace("'", "")) 


# In[1347]:


# 최종 라벨 및 bbox 좌표

labels


# ---

# ## 1.2 Langchain :  new object bounding box extraction

# ---

# In[1372]:


from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser


class CommaSeparatedListOutputParser(BaseOutputParser):
    """LLM 아웃풋에 있는 ','를 분리해서 리턴하는 파서."""


    def parse(self, text: str):
        return text.strip().split(", ")


template = """
Let’s imagine you’re a semantic parser. I’ll provide specific bounding box coordinates and
instructions. These coordinates represent the [(left corner x, y values), (right corner x, y values)]
of the bounding box. Extract the action and target from the instructions and new bounding
box coordinates, noting the association between bounding box coordinates and the target. Also,
bounding box coordinates should not exceed the frame size [(0,0), (850,425)].  Then, adjust the
bounding box using expected values and display the resulting coordinates as the output. Let’s think step by step.

condition : 
First, please maintain the width-to-height ratio of the box.
Second, unless the object is moving, please execute the insturction with the left corner x, y values of the box fixed.
Third, represent the coordinates as integers.

For example:
 
insturction = The pillow located on the far right is reduced in area and moved to the left of the sofa

Bounding Box Coordinates: [['sofa_1 [174 185 604 323]',
 'table_1 [289 288 489 365]',
 'pillow_2 [427 188 493 248]',
 'pillow_1 [230 185 285 246]',
 'table_2 [ 75 232 169 314]',
 'pillow_3 [282 187 345 249]']]
 
output = [
    'action': 'the pillow located on the far right is reduced in area','target': 'pillow_2',  nbox': [(427 188 467 218)],
    'action': 'moved to the left of the sofa','target': 'pillow_2', 'nbox': [(174 185 214 215)]
]"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)

human_template = """
Instruction : {instruction}. Please remind the condition.

Bounding Box Coordinates: {bounding_box_coordinates}

The output is returned in the form of a dictionary.
Output:"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])
chain = chat_prompt | ChatOpenAI(openai_api_key = "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", temperature=1) | CommaSeparatedListOutputParser()
output= chain.invoke({"instruction": question, "bounding_box_coordinates": ','.join(labels)})


# In[ ]:


# 수행된 명령어 및 llm으로 보내준 bbox 값 좌표 출력용 
{"instruction": question, "bounding_box_coordinates": ','.join(labels)}


# In[1375]:


# llm_2 결과 output 출력 
print(output)
print(type(output))


# In[1377]:


# 리스트를 개행문자(" "")로 연결하여 문자열로 변환
formatted_text_string = " ".join(output)

print(formatted_text_string)
print(type(formatted_text_string))


# In[1379]:


input_string = formatted_text_string


# 정규 표현식을 사용하여 target 값과 nbox 값 추출
targets = re.findall(r"'target': '(.*?)'", input_string)
nbox_values = re.findall(r"\(([\d.]+ [\d.]+ [\d.]+ [\d.]+)\)", input_string)

# nbox 값을 실수로 변환하고 리스트로 구성
nbox_list = [[float(coord) for coord in box.split()] for box in nbox_values]

# 결과 딕셔너리 생성
result = [{'target': targets[0], 'nbox': nbox_list[0]},
          {'target': targets[1], 'nbox': nbox_list[1]}]

# 결과 출력
print(result)


# In[1380]:


# # # # # parser용
# from langchain.output_parsers import CommaSeparatedListOutputParser

# output_parser = CommaSeparatedListOutputParser()

# object_names = output_parser.parse(formatted_text_string)
# object_names


# 출력된 target 값과 new bbox 값 이미지 frame상에 재표시 

# In[1382]:


from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import torch
from PIL import Image
from torchvision.ops import box_convert
import bisect

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap


# In[1383]:


phrase_new = []
boxes_new = []
for i in range(len(result)):
    phrase_new.append(result[i]['target'])
    boxes_new.append(result[i]['nbox'])
print(phrase_new,'\n', boxes_new)
print(type(phrase_new),'\n', type(boxes_new))


# In[1385]:


def annotate_new(image_source: np.ndarray, boxes: torch.Tensor, phrases: List[str]) -> np.ndarray:
    
    detections = sv.Detections(xyxy=boxes)
    print('det', detections)

    # labels = [
    #     f"{phrase} {logit:.2f}"
    #     for phrase, logit
    #     in zip(phrases, logits)
    # ]

    box_annotator = sv.BoxAnnotator()
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=phrases)
    return annotated_frame


# In[1387]:


boxes_new = boxes_new
phrase_new = phrase_new
boxes_new = torch.tensor(boxes_new)
# print(boxes_new)
# print(phrase_new)
logits = []
for i in range(len(phrase_new)):
    logits.append(0.9)
    
logits = torch.tensor(logits)   
# print(logits)

annotated_frame = annotate_new(image_source=image_source, boxes=boxes_new.numpy(), phrases=phrase_new)

# get_ipython().run_line_magic('matplotlib', 'inline')
# sv.plot_image(annotated_frame, (10, 10))
cv2.imwrite(f"./outputs/{now}/{IMAGE_NAME.split('.')[0]}_final_output.jpg", annotated_frame)


# ### 추가하면 좋은 사항 
# #### 후보 1. captioning을 받아서 자동으로 여러 text augmentation 시키기.
# #### 후보 2. 물체가 왼쪽을 바라보고 있는지 등... pose(viewpoint)를 반영시킬 수 있을까?
# #### 후보 3. depth 정보를 받으면 멀리 있는 위치도 바꿀 수 있을 듯 

# 
