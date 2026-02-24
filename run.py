import os
import glob
from pathlib import Path
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.cli.predict import predict_image
from sharp.utils.gaussians import save_ply

def main():
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    checkpoint_path = Path("model/sharp_2572gikvuh.pt")

    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist.")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Get supported extensions
    extensions = io.get_supported_image_extensions()
    
    # Find all images in input_dir
    image_paths = []
    for ext in extensions:
        image_paths.extend(list(input_dir.glob(f"**/*{ext}")))

    if len(image_paths) == 0:
        print("No valid images found in data/input.")
        return

    # Filter out already processed images
    unprocessed_images = []
    for img_path in image_paths:
        expected_output = output_dir / f"{img_path.stem}.ply"
        if not expected_output.exists():
            unprocessed_images.append(img_path)

    if len(unprocessed_images) == 0:
        print("All images have already been processed.")
        return

    print(f"Found {len(unprocessed_images)} unprocessed image(s).")

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)

    # Process images
    for image_path in unprocessed_images:
        print(f"Processing {image_path.name}...")
        
        # Load image and focal length
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        
        # Predict 3D Gaussians
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))
        
        # Save PLY file
        output_file = output_dir / f"{image_path.stem}.ply"
        print(f"Saving 3DGS to {output_file}")
        save_ply(gaussians, f_px, (height, width), output_file)

    print("All processing completed!")

if __name__ == "__main__":
    main()
