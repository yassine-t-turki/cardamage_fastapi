import os
from PIL import Image
from config import UPLOAD_DIR
import supervision as sv
from typing import List


# Helper function to get list of uploaded files
def get_uploaded_files() -> List[str]:
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

# Helper function to get sv.Position from string
def get_position_from_string(position: str) -> sv.Position:
    positions = {
        "CENTER": sv.Position.CENTER,
        "CENTER_LEFT": sv.Position.CENTER_LEFT,
        "CENTER_RIGHT": sv.Position.CENTER_RIGHT,
        "TOP_CENTER": sv.Position.TOP_CENTER,
        "TOP_LEFT": sv.Position.TOP_LEFT,
        "TOP_RIGHT": sv.Position.TOP_RIGHT,
        "BOTTOM_LEFT": sv.Position.BOTTOM_LEFT,
        "BOTTOM_CENTER": sv.Position.BOTTOM_CENTER,
        "BOTTOM_RIGHT": sv.Position.BOTTOM_RIGHT
    }
    return positions.get(position.upper(), sv.Position.TOP_RIGHT)  # Default to TOP_RIGHT

def resize_image(image, base_height=None, base_width=None):
    """
    Resize the image while keeping the aspect ratio.
    """
    if base_height:
        aspect_ratio = base_height / float(image.height)
        new_width = int(aspect_ratio * image.width)
        new_size = (new_width, base_height)
    elif base_width:
        aspect_ratio = base_width / float(image.width)
        new_height = int(aspect_ratio * image.height)
        new_size = (base_width, new_height)
    else:
        return image

    return image.resize(new_size)

def combine_images(image1, image2, align='horizontal'):
    """
    Combine two images either horizontally or vertically.
    """
    if align == 'horizontal':
        # Resize images to the same height
        height1, height2 = image1.height, image2.height
        max_height = max(height1, height2)

        image1_resized = resize_image(image1, base_height=max_height)
        image2_resized = resize_image(image2, base_height=max_height)

        # Create a new image with combined width
        combined_width = image1_resized.width + image2_resized.width
        combined_image = Image.new('RGB', (combined_width, max_height))

        # Paste the images into the combined image
        combined_image.paste(image1_resized, (0, 0))
        combined_image.paste(image2_resized, (image1_resized.width, 0))

    elif align == 'vertical':
        # Resize images to the same width
        width1, width2 = image1.width, image2.width
        max_width = max(width1, width2)

        image1_resized = resize_image(image1, base_width=max_width)
        image2_resized = resize_image(image2, base_width=max_width)

        # Create a new image with combined height
        combined_height = image1_resized.height + image2_resized.height
        combined_image = Image.new('RGB', (max_width, combined_height))

        # Paste the images into the combined image
        combined_image.paste(image1_resized, (0, 0))
        combined_image.paste(image2_resized, (0, image1_resized.height))

    return combined_image