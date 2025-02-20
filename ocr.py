from PIL import Image
import pytesseract

# Load the image from file
image_path = 'path_to_your_image.jpg'
image = Image.open(image_path)

# Use pytesseract to extract text
extracted_text = pytesseract.image_to_string(image)

# Print the extracted text
print(extracted_text)
