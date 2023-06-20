import argparse
from PIL import Image
import torch
from pipeline_gradop_stroke2img import GradOPStroke2ImgPipeline

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pipeline with Stable Diffusion Weights
    pipeline = GradOPStroke2ImgPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float32).to(device)
    
    # Load the user-scribbles image
    stroke_img = Image.open(args.img_path).convert('RGB').resize((512,512))
    
    # Define the generator
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    if args.method == 'gradop+':
        # Perform img2img guided synthesis using gradop+
        out = pipeline.gradop_plus_stroke2img(args.prompt, stroke_img, strength=args.strength, num_iterative_steps=args.num_iterative_steps, grad_steps_per_iter=args.grad_steps_per_iter, generator=generator)
    elif args.method == 'sdedit':
        # Perform img2img predictions using SDEdit-based diffusers
        out = pipeline.sdedit_img2img(prompt=args.prompt, image=stroke_img, generator=generator)
    
    # Construct output image name if not provided
    if args.output_img is None:
        args.output_img = f"./output-images/{args.prompt.replace(' ', '-')}_{args.method}_{args.seed}.png"
    
    # Save the output image
    out.save(args.output_img)

    print(f"Your output image is at {args.output_img}... Enjoy!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GradOP+ and SDEdit Image Generation')
    parser.add_argument('--method', type=str, choices=['gradop+', 'sdedit'], default='gradop+', help='Method type: "gradop+" or "sdedit"')
    parser.add_argument('--img_path', type=str, default='./input-images/fox.png', help='Path to the input image')
    parser.add_argument('--prompt', type=str, default='a photo of a fox beside a tree', help='Text prompt for the image generation')
    parser.add_argument('--seed', type=int, default=0, help='Seed for randomness')
    parser.add_argument('--strength', type=float, default=0.8, help='Strength for the GradOP+ method')
    parser.add_argument('--output_img', type=str, help='Path to the output image')
    parser.add_argument('--num_iterative_steps', type=int, default=3, help='Number of iterative steps for GradOP+')
    parser.add_argument('--grad_steps_per_iter', type=int, default=12, help='Number of gradient steps per iterative step for GradOP+')

    args = parser.parse_args()
    main(args)
