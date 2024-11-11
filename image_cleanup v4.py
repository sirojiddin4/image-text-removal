import io
import os
import re
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

# Define paths for input and output directories
source_folder = r'C:\Users\user\Desktop\Thesis\dataset\memotion_dataset_7k\images'
output_folder = r'C:\Users\user\Desktop\Thesis\output_images_60'
os.makedirs(output_folder, exist_ok=True)

# Match all files starting with "image_" and ending with a number between 101 and 200, regardless of extension
image_files = sorted(
    [f for f in os.listdir(source_folder) if re.match(r'image_(6[0-9][0-9][0-9]|6992)\..+$', f, re.IGNORECASE)]
)

def process_images_in_batches(image_files, source_folder, output_folder):
    for image_file in image_files:
        input_path = os.path.join(source_folder, image_file)
        output_path = os.path.join(output_folder, image_file)
        
        try:
            # Open the image with PIL and convert to RGB to avoid palette limit issues
            pil_image = Image.open(input_path).convert("RGB")
        except (IOError, OSError) as e:
            print(f"Unable to open or process image {input_path}. Error: {e}")
            continue

        # Prepare image for Google Vision API
        with io.open(input_path, 'rb') as image_file:
            content = image_file.read()
        image = vision.Image(content=content)

        # Perform text detection
        response = client.text_detection(image=image)
        annotations = response.text_annotations

        # Draw over detected text areas
        draw = ImageDraw.Draw(pil_image)
        for text in annotations[1:]:  # Skip the first annotation (the full text)
            vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            draw.polygon(vertices, fill="black")

        # Save the modified image
        try:
            pil_image.save(output_path)
        except (IOError, OSError) as e:
            print(f"Failed to save modified image {output_path}. Error: {e}")

        # Handle any API errors
        if response.error.message:
            raise Exception(f'Error processing image {image_file}: {response.error.message}')

# Run the function to process images 101-200
process_images_in_batches(image_files, source_folder, output_folder)

print("Processing complete. Modified images are saved in:", output_folder)
