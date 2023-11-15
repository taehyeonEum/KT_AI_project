import argparse
import grounded_sam_simple_demo_2_langchain as gpt_grouded_sam
import in_painting_with_stable_diffusion_using_diffusers as in_painting


def main(image_name):

    print("\n\n\n***************************************")
    print("in main gpt_grounded_sam process...")
    results = gpt_grouded_sam.gpt_grounded_sam(image_name)
    for i, result in enumerate(results):
        print("\n\n\n------------------------------------")
        print(f"in main {i}th inpainting process...")
        print(f"result: {result}")
        in_painting.inpainting(i, result)
    
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_name", type=str)

    args = parser.parse_args()

    main(args.image_name)