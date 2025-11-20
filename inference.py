import random
from PIL import Image, ImageDraw
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Specify the actual path to the saved generator model
generator_model_path = "/path/to/your/saved/generator.pth"

# Load the saved generator model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator()  # Replace with the actual generator class name if it's different
generator.load_state_dict(torch.load(generator_model_path, map_location=device))
generator = generator.to(device)

# Function to create a circular mask
def create_random_circular_mask(img_size=(64, 64), radius=7):
    """Creates a circular mask with random position on an image of size img_size."""
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    x, y = random.randint(radius, img_size[0] - radius), random.randint(radius, img_size[1] - radius)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Shape: [1, 1, H, W]
    return mask

# Function to test the model on an unseen image
def test_on_unseen_image(image_path, generator, transform, device="cpu"):
    # Load and preprocess unseen image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Shape: [1, 3, 64, 64]

    # Create a random circular mask and apply it
    mask = create_random_circular_mask(img_size=(64, 64)).to(device)  # Shape: [1, 1, 64, 64]
    masked_image = image * (1 - mask)  # Apply mask to the image

    generator.eval()
    with torch.no_grad():
        inpainted_image = generator(masked_image, mask)  
    # Denormalize images for visualization
    image = (image * 0.5 + 0.5).cpu()
    masked_image = (masked_image * 0.5 + 0.5).cpu()
    inpainted_image = (inpainted_image * 0.5 + 0.5).cpu()

    # Plot the original, masked, and inpainted images
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[0].permute(1, 2, 0))
    axes[0].set_title("Original Image")
    axes[1].imshow(masked_image[0].permute(1, 2, 0))
    axes[1].set_title("Masked Image")
    axes[2].imshow(inpainted_image[0].permute(1, 2, 0))
    axes[2].set_title("Inpainted Image")

    for ax in axes:
        ax.axis("off")
    plt.show()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Test on an unseen image
test_image_path = "/content/drive/MyDrive/ABHAY_5.jpeg"
test_on_unseen_image(test_image_path, generator, transform, device=device)
