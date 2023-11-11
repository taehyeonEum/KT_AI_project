
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# OUTPUT_DIR = "./output_thex"

# def create_image_with_bounding_box(bbox):
#     # 이미지 크기 설정
#     image_size = (512, 512)

#     # 넘파이 배열 생성 (전체를 0으로 초기화)
#     image = np.zeros(image_size)

#     # bounding box 좌표
#     x_min, y_min, x_max, y_max = bbox

#     # bounding box 안의 영역을 256으로 설정
#     image[y_min:y_max, x_min:x_max] = 256

#     return image

# bounding_box_coordinates = (100, 100, 300, 300)

# # 이미지 생성
# image_with_bbox = create_image_with_bounding_box(bounding_box_coordinates)

# cv2.imwrite(os.path.join(OUTPUT_DIR, "image_with_bbox.png"), image_with_bbox)


arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8], 
                [9, 10, 11, 12], 
                [13, 14, 15, 16]
                ])

arr_cropped = arr[1:3, 1:3]
print(arr_cropped)
print((arr_cropped.shape))