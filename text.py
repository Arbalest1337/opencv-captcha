import cv2
import easyocr


def ocr_get_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    reader = easyocr.Reader(["ch_sim"], gpu=False)
    results = reader.readtext(binary)
    text = ""
    for result in results:
        text += result[1]
    return text
