import pytesseract
from PIL import Image
import requests
from io import BytesIO
import os

if not os.getenv("TESSERACT_PATH"):
    raise Exception("TESSERACT_PATH environment variable is not set")

pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_PATH")

def get_file_extension(url) -> str:
    # remove query parameters if present
    url = url.split("?")[0]
    return os.path.splitext(url)[1].lower()

def clean_text(text):
    # Remove newlines and extra spaces and lines with only numbers or less than 3 words
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        if len(line.split()) < 3:
            continue
        if all(word.isdigit() for word in line.split()):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def process_image(image_url) -> str:
    try:
        response = requests.get(image_url, stream=True)
        image = Image.open(BytesIO(response.content))
        image = image.convert("L")

        custom_config = r"--oem 3 --psm 4"

        text = pytesseract.image_to_string(image, config=custom_config)

    except Exception as e:
        print(f"Error processing image from {image_url}: {e}")
        return ""

    return clean_text(text)
