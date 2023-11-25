import grounded_sam_simple_demo_2_langchain as gpt_grouded_sam
import in_painting_with_stable_diffusion_using_diffusers as inpainting
from InpaintAnything import remove_anything as remove_anything


if __name__=="__main__":

    # 마지막 실험 번호, question 읽어오기
    f=open("./txt_s/ex_number_question_series.txt", "r")
    lines = f.readlines()
    f.close()

    ex_num = 1

    for line in lines:

        image_name = line.split(": ")[0]
        question = line.split(": ")[1]
        # 번호 지정하여 코드 실행하는 코드
        try: 
            ex_num = int(line.split(": ")[2])
            try:
                print("\n\n\n***************************************")
                print("***************************************")
                print("***************************************")
                print(f"experiment number : {ex_num}")
                print(f"experiment question : {question}")
                print("in main gpt_grounded_sam process...")
                results = gpt_grouded_sam.gpt_grounded_sam(image_name, ex_num, question)

                for i, result in enumerate(results):
                    print("\n\n\n------------------------------------")
                    print(f"in main {i}th inpainting process...")
                    print(f"result: {result}")
                    inpainting.inpainting(i, result, ex_num)
                    
            except Exception as e:
                f=open("./txt_s/error.txt", 'a')
                f.write(f"ex_num: {ex_num}_image_name: {image_name}_question: {question}_error: {e}\n")
                f.close()
                print(e)

        except Exception as e:
            pass

        
        ex_num = ex_num + 1


    # # 다음 실험 번호 자동 추가
    # f=open("./ex_number_question.txt", 'a')
    # f.write("\n"+str(ex_num+1))
    # f.close()