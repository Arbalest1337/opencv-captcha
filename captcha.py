import cv2
import numpy as np
import easyocr


def ocr_get_text(image_path):
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 执行腐蚀操作
    eroded_image = cv2.erode(binary, kernel, iterations=2)
    cv2.imshow("Hollow Shapes", eroded_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

    reader = easyocr.Reader(["ch_sim"], gpu=False)
    results = reader.readtext(image)
    text = ""
    for result in results:
        text += result[1]
    return text


image_path = "images/2.png"
question = ocr_get_text(image_path)
print(question)
