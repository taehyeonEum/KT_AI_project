KT_project : Inri_AI

실행방법: 
CUDA_VISIBLE_DEVICES=0 python main.py --image_name {image_name}
예시) CUDA_VISIBLE_DEVICES=0 python main.py --image_name img_0.jpeg


실험 소스(대상이미지)는 ./content에 저장되어 있음.

!! 모든 실험 결과는 EX_{index}에 폴더 별로 저장. 

실험 input은 ./ex_number_image_question.txt에 저장됨. 코드에서는 가장 마지막 라인의 image에 question을 적용함. 
(위와 같이 구성한 이유는 실험을 진행할 때 효율을 추구하기 위함. 실험 로그가 자연스럽게 남게 됩니다.)

dino, grounded_sam 결과 : outputs_grounded_sam 폴더에 저장
  폴더 구성은 다음과 같음. 
  EX_{number}/{image_name}/{object_id}/masks.... # 여러가지 필요한 mask 저장.

inpainting 결과 : outputs_inpaintings 폴더에 저장
  폴더 구성은 다음과 같음.
  EX_{number}/{image_name}/{object}/{action_id}/
    {diffusion result # diffusion 결과 3가지 사진 },
    {object_removed_image # 배경에서 물체가 지워진 사진},
    {original_image # 비교가 용이하게 하기 위해서 512x512의 원본 사진을 저장함},
    {remove_object_square_mask # 배경을 잘 지우기 위해 생성한 마스크.}
