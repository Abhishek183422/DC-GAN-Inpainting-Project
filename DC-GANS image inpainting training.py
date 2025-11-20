import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import numpy as np
import random
import io
import boto3
# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = 'maxillofacial-prostheses'
s3_folder = 'celeba_hq/'
class CelebAInpaintingDataset(Dataset):
    def __init__(self, bucket, folder, transform=None, mask_radius=7, img_size=(64, 64)):
        self.bucket = bucket
        self.folder = folder
        self.transform = transform
        self.mask_radius = mask_radius
        self.img_size = img_size
       
        # Get list of all images in the specified S3 folder
        self.image_paths = []
        response = s3.list_objects_v2(Bucket=self.bucket, Prefix=self.folder)
        if 'Contents' in response:
            for obj in response['Contents']:
                if obj['Key'].endswith(('.png', '.jpg', '.jpeg')): # Filter for image files
                    self.image_paths.append(obj['Key'])
   
    def create_circular_mask(self, height, width):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        x, y = random.randint(0, width), random.randint(0, height)
        draw.ellipse((x - self.mask_radius, y - self.mask_radius, x + self.mask_radius, y + self.mask_radius), fill=255)
        return np.array(mask)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        img_key = self.image_paths[idx]
       
        # Load image directly from S3
        response = s3.get_object(Bucket=self.bucket, Key=img_key)
        image = Image.open(io.BytesIO(response['Body'].read())).convert("RGB").resize(self.img_size)
       
        mask = self.create_circular_mask(*self.img_size)
       
        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
       
        masked_image = image * (1 - mask)
        return image, masked_image, mask
# Define transformations and create DataLoader
transform = transforms.Compose([
    transforms.Resize((64, 64)), # here we are setting the size of the image to 64*64 RGB
    transforms.ToTensor(),
])
dataset = CelebAInpaintingDataset(bucket=bucket_name, folder=s3_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Hyperparameters
lr = 0.0002 # can also tried with 0.0001 but this was working
num_epochs = 1000 # can increase it to more epoch
beta1 = 0.5 # DCGAN beta1 parameter for Adam optimizer the higher better, 0.5 seems working
# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)
# Optimizers
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
# Loss function
criterion = nn.BCELoss()
# Labels for real and fake images
real_label = 1.
fake_label = 0.
# Training Loop
for epoch in range(num_epochs):
    for i, (real_image, masked_image, mask) in enumerate(dataloader):
        # Move data to GPU if available
        real_image = real_image.to(device)
        masked_image = masked_image.to(device)
        mask = mask.to(device)
        # Train Discriminator
        discriminator.zero_grad()
        label = torch.full((real_image.size(0),), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through Discriminator
        output = discriminator(real_image).view(-1)
        lossD_real = criterion(output, label)
        lossD_real.backward()
        # Forward pass fake batch through Discriminator
        fake_image = generator(masked_image, mask)
        label.fill_(fake_label)
        output = discriminator(fake_image.detach()).view(-1)
        lossD_fake = criterion(output, label)
        lossD_fake.backward()
        # Update Discriminator
        optimizerD.step()
        lossD = lossD_real + lossD_fake
        # Train Generator
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_image).view(-1)
        lossG = criterion(output, label)
        lossG.backward()
        optimizerG.step()
        # Print losses every 50 batches
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss D: {lossD.item()}, Loss G: {lossG.item()}")
    # Visualization every 10 epochs
    if epoch % 10 == 0:
        with torch.no_grad():
            fake_image = generator(masked_image, mask)
            fake_image = (fake_image * 0.5 + 0.5).cpu().numpy() # Denormalize for visualization
            real_image = (real_image * 0.5 + 0.5).cpu().numpy()
            masked_image = (masked_image * 0.5 + 0.5).cpu().numpy()
            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(real_image[0].transpose(1, 2, 0))
            axes[0].set_title("Original Image")
            axes[1].imshow(masked_image[0].transpose(1, 2, 0))
            axes[1].set_title("Masked Image")
            axes[2].imshow(fake_image[0].transpose(1, 2, 0))
            axes[2].set_title("Inpainted Image")
            plt.show()
