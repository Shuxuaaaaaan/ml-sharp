import os
import time
from pathlib import Path
import torch

from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.cli.predict import predict_image
from sharp.utils.gaussians import save_ply

try:
    from rich.console import Console
    from rich.progress import track
    console = Console()
except ImportError:
    class Console:
        def print(self, msg, *args, **kwargs):
            print(msg)
    console = Console()
    def track(sequence, description=""):
        return sequence

def main():
    input_dir = Path("data/input")
    output_dir = Path("data/output")
    checkpoint_path = Path("model/sharp_2572gikvuh.pt")

    if not input_dir.exists():
        console.print(f"[bold red]Input directory {input_dir} does not exist.[/]")
        return

    output_dir.mkdir(exist_ok=True, parents=True)

    # Get supported extensions
    extensions = io.get_supported_image_extensions()
    
    # Find all images in input_dir and remove duplicates (due to case-insensitive globbing on Windows)
    image_paths_set = set()
    for ext in extensions:
        for p in input_dir.glob(f"**/*{ext}"):
            image_paths_set.add(p.resolve())
    
    image_paths = list(image_paths_set)

    if len(image_paths) == 0:
        console.print("[yellow]No valid images found in data/input.[/]")
        return

    # Filter out already processed images
    unprocessed_images = []
    for img_path in image_paths:
        expected_output = output_dir / f"{img_path.stem}.ply"
        if not expected_output.exists():
            unprocessed_images.append(img_path)

    if len(unprocessed_images) == 0:
        console.print("[bold green]All images have already been processed![/]")
        return

    console.print(f"[bold cyan]Found {len(unprocessed_images)} unprocessed image(s).[/]")

    # Setup device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    console.print(f"Using device: [bold yellow]{device}[/]")

    # Load model
    console.print(f"Loading checkpoint from [magenta]{checkpoint_path}[/]...")
    
    start_load = time.time()
    state_dict = torch.load(checkpoint_path, weights_only=True)
    gaussian_predictor = create_predictor(PredictorParams())
    gaussian_predictor.load_state_dict(state_dict)
    gaussian_predictor.eval()
    gaussian_predictor.to(device)
    load_time = time.time() - start_load
    console.print(f"[green]Model loaded in {load_time:.2f} seconds.[/]\n")

    # Process images
    total_start_time = time.time()
    
    for i, image_path in enumerate(unprocessed_images, 1):
        console.print(f"[bold blue][{i}/{len(unprocessed_images)}][/] Processing [bold]{image_path.name}[/]...")
        img_start_time = time.time()
        
        # Load image and focal length
        image, _, f_px = io.load_rgb(image_path)
        height, width = image.shape[:2]
        
        # Predict 3D Gaussians
        gaussians = predict_image(gaussian_predictor, image, f_px, torch.device(device))
        
        # Save PLY file
        output_file = output_dir / f"{image_path.stem}.ply"
        save_ply(gaussians, f_px, (height, width), output_file)
        
        img_time = time.time() - img_start_time
        console.print(f"  [green]✓ Saved to[/] {output_file.name} [dim](took {img_time:.2f}s)[/]\n")

    total_time = time.time() - total_start_time
    avg_time = total_time / len(unprocessed_images)
    console.print(f"[bold green]✨ All processing completed![/] Total time: {total_time:.2f}s (Avg: {avg_time:.2f}s/image)")

if __name__ == "__main__":
    main()
