import pytesseract
import PIL
from PIL import Image
img=Image.open('02.jpg')
text=pytesseract.img_to_string(img)
print(text)