import grounded_sam_simple_demo_2_langchain as gpt_grouded_sam
import Gradio_in_painting_with_stable_diffusion_using_diffusers as inpainting
from InpaintAnything import remove_anything as remove_anything
import os
import cv2

# if __name__=="__main__":

    # 마지막 실험 번호, question 읽어오기
    # f=open("./txt_s/EX_NUMber_question_series3.txt", "r")
    # lines = f.readlines()
    # f.close()

INPUT_DIR = "./Gradio/content"
INDEX_PATH = "./Gradio/gradio_index.txt"
GROUNDED_SAM_OUTPUT_DIR = "./Gradio/outputs_grounded_sam_oneshot"
INPAINTING_OUTPUT_DIR = "./Gradio/outputs_inpainting_oneshot"
OPENAI_API = "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" #key : yyy
QUANTITATIVE_LOG = "./Gradio/outputs_grounded_sam_oneshot/quantitative_log.txt"
os.makedirs(GROUNDED_SAM_OUTPUT_DIR, exist_ok=True)
os.makedirs(INPAINTING_OUTPUT_DIR, exist_ok=True)


def run(input_image, question):
    # f = open(os.path.join(INPAINTING_OUTPUT_DIR, '_'+question), 'w')
    # f.close()

    f = open(INDEX_PATH, 'r')
    EX_NUM = int(f.read()) +1
    f.close()

    print("EX_NUM: ", EX_NUM)

    f =open(INDEX_PATH, 'w')
    f.write(str(EX_NUM))
    f.close()

    cv2.imwrite(os.path.join(INPUT_DIR, f"input_image.jpeg"), cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
    # ex_num = 1

    f = open(QUANTITATIVE_LOG, 'w')
    f.close()

    # for line in lines:

    f = open(QUANTITATIVE_LOG, 'a')
    f.write(f"EX_{EX_NUM}\n")
    f.close()

    # image_name = line.split(": ")[0]
    # question = line.split(": ")[1]
    # 번호 지정하여 코드 실행하는 코드
    # try: 
    #     EX_NUM = int(line.split(": ")[2])
    try:
        while_count = 0
            
        print("\n\n\n***************************************")
        print("***************************************")
        print("***************************************")
        print(f"experiment number : {EX_NUM}")
        print(f"experiment question : {question}")
        print("in main gpt_grounded_sam process...")
        while True:
            is_done, results = gpt_grouded_sam.gpt_grounded_sam("input_image.jpeg", EX_NUM, question, INPUT_DIR, GROUNDED_SAM_OUTPUT_DIR, OPENAI_API, QUANTITATIVE_LOG)
            if is_done:
                break
            while_count = while_count + 1

            
        print(f"while count: {while_count}")
        for i, result in enumerate(results):
            print("\n\n\n------------------------------------")
            print(f"in main {i}th inpainting process...")
            print(f"result: {result}")
            is_done, outputs = inpainting.inpainting(i, result, EX_NUM, INPUT_DIR, GROUNDED_SAM_OUTPUT_DIR, INPAINTING_OUTPUT_DIR)
        

    except Exception as e:
        f=open("./txt_s/error_sppliments.txt", 'a')
        f.write(f"EX_NUM: {EX_NUM}_question: {question}_error: {e}\n")
        f.close()
        print(e)
        # EX_NUM = EX_NUM - 1

    # except Exception as e:
    #     pass

    
    # EX_NUM = EX_NUM + 1
    return outputs


import gradio as gr

with gr.Blocks() as iface:
  gr.HTML("<h1 style='text-align: center;'>인리AI Demo</h1>")
  gr.Markdown("Hello, sir. Upload your image and Enter the instructions you want!")
  with gr.Row():
    input_image = gr.Image()
    input_text = gr.Textbox(placeholder="Example: Make TV bigger.")

  btn = gr.Button("Submit")
  gr.Markdown("These are the suggested results :)")
  with gr.Row():
    out = [gr.Image(), gr.Image(), gr.Image()]

  # 버튼 클릭시 greet를 호출하고, inp에 입력된 문자열을 파라미터로 보낸다.
  # 함수의 반환값은 out에 출력한다.
  btn.click(fn=run, inputs=[input_image, input_text], outputs=out)

iface.launch(debug=False, server_name="0.0.0.0", server_port=7860)