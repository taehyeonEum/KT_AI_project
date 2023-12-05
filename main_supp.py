import grounded_sam_simple_demo_2_langchain as gpt_grouded_sam
import in_painting_with_stable_diffusion_using_diffusers as inpainting
from InpaintAnything import remove_anything as remove_anything
import os


if __name__=="__main__":

    # 마지막 실험 번호, question 읽어오기
    f=open("./txt_s/ex_number_question_series3.txt", "r")
    lines = f.readlines()
    f.close()

    ex_num = 1
    INPUT_DIR = "./content"
    GROUNDED_SAM_OUTPUT_DIR = "./outputs_grounded_sam_oneshot"
    INPAINTING_OUTPUT_DIR = "./outputs_inpainting_oneshot"
    OPENAI_API = "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" #key ddd
    QUANTITATIVE_LOG = "./outputs_grounded_sam_oneshot/quantitative_log.txt"
    # os.makedirs(GROUNDED_SAM_OUTPUT_DIR, exist_ok=True)
    # os.makedirs(INPAINTING_OUTPUT_DIR, exist_ok=True)

    f = open(QUANTITATIVE_LOG, 'w')
    f.close()

    exist_dir = os.listdir(INPAINTING_OUTPUT_DIR)
    exist_idx = []
    for dir_name in exist_dir:
        idx = int(dir_name.split("_")[-1])
        exist_idx.append(idx)

    total_idxes = list(range(len(lines)))
    missing_idxes = [x for x in total_idxes if x not in exist_idx]
    print("total_indexs: \n", total_idxes)
    print("exist_indexs: \n", exist_idx)
    print("missing_indexs: \n", missing_idxes)


    for idx, line in enumerate(lines):

        ex_num = idx + 1
        if ex_num not in exist_idx:

            f = open(QUANTITATIVE_LOG, 'a')
            f.write(f"EX_{ex_num}\n")
            f.close()

            image_name = line.split(": ")[0]
            question = line.split(": ")[1]
            # 번호 지정하여 코드 실행하는 코드
            # try: 
            #     ex_num = int(line.split(": ")[2])
            try:
                while_count = 0
                    
                print("\n\n\n***************************************")
                print("***************************************")
                print("***************************************")
                print(f"experiment number : {ex_num}")
                print(f"experiment question : {question}")
                print("in main gpt_grounded_sam process...")
                while True:
                    is_done, results = gpt_grouded_sam.gpt_grounded_sam(image_name, ex_num, question, INPUT_DIR, GROUNDED_SAM_OUTPUT_DIR, OPENAI_API, QUANTITATIVE_LOG)
                    if is_done:
                        break
                    while_count = while_count + 1

                    
                print(f"while count: {while_count}")
                for i, result in enumerate(results):
                    print("\n\n\n------------------------------------")
                    print(f"in main {i}th inpainting process...")
                    print(f"result: {result}")
                    is_done = inpainting.inpainting(i, result, ex_num, INPUT_DIR, GROUNDED_SAM_OUTPUT_DIR, INPAINTING_OUTPUT_DIR)
                

            except Exception as e:
                f=open("./txt_s/error_sppliments.txt", 'a')
                f.write(f"ex_num: {ex_num}_image_name: {image_name}_question: {question}_error: {e}\n")
                f.close()
                print(e)
                # ex_num = ex_num - 1

            # except Exception as e:
            #     pass

        

    # # 다음 실험 번호 자동 추가
    # f=open("./ex_number_question.txt", 'a')
    # f.write("\n"+str(ex_num+1))
    # f.close()