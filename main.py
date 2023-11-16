import argparse
import grounded_sam_simple_demo_2_langchain as gpt_grouded_sam
import in_painting_with_stable_diffusion_using_diffusers as inpainting


def main(image_name, mode):

    print("\n\n\n***************************************")
    print("in main gpt_grounded_sam process...")
    results = gpt_grouded_sam.gpt_grounded_sam(image_name, mode)

    for i, result in enumerate(results):
        print("\n\n\n------------------------------------")
        print(f"in main {i}th inpainting process...")
        print(f"result: {result}")
        inpainting.inpainting(i, result)
    
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--image_name", type=str)
    parser.add_argument("--mode", type=str) # mode : 'bigger', 'smaller', 'move'

    args = parser.parse_args()

    main(args.image_name, args.mode)