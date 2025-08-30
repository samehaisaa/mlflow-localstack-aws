import os
from PIL import Image
import random

def create_sample_image(path, color):
    """Create a simple colored square image."""
    img = Image.new('RGB', (224, 224), color=color)
    # Add some noise to make images different
    pixels = img.load()
    for i in range(50):
        x = random.randint(0, 223)
        y = random.randint(0, 223)
        pixels[x, y] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    img.save(path)

# Create sample images for cats (blueish)
for i in range(5):
    create_sample_image(f"data/train/cat/cat_{i}.jpg", (100, 100, 200))
    if i < 2:
        create_sample_image(f"data/val/cat/cat_val_{i}.jpg", (100, 100, 200))

# Create sample images for dogs (reddish)
for i in range(5):
    create_sample_image(f"data/train/dog/dog_{i}.jpg", (200, 100, 100))
    if i < 2:
        create_sample_image(f"data/val/dog/dog_val_{i}.jpg", (200, 100, 100))

print("Sample images created!")
