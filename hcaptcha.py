from text import ocr_get_text
from shape import get_shapes
import re
import cv2


def str2vertices(input_string):
    if re.search(r"三|角", input_string):
        return 3
    elif re.search(r"四|正|长|菱", input_string):
        return 4
    elif re.search(r"星", input_string):
        return 5
    elif re.search(r"六", input_string):
        return 6
    else:
        return 0


image_path = "images/hcaptcha.png"
question = ocr_get_text(image_path)
print(question)

outer_shape_str = question.split("内的")[0]
inner_shape_str = question.split("内的")[1]
outer_vertices = str2vertices(outer_shape_str)
inner_vertices = str2vertices(inner_shape_str)

shapes, image = get_shapes(image_path)

result = next(
    (
        item
        for item in shapes
        if item["parent_vertices"] == outer_vertices
        and item["vertices"] == inner_vertices
    ),
    None,
)

print(result["x"], result["y"])


# show result
for shape in shapes:
    cv2.drawContours(image, shape["parent_contour"], -1, (0, 255, 0), 3)
    cv2.drawContours(image, shape["contour"], -1, (0, 0, 255), 3)
    cv2.putText(
        image,
        f"{shape['vertices']}",
        (shape["x"] - 8, shape["y"] + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 200),
        2,
    )
    cv2.putText(
        image,
        f"{shape['parent_vertices']}",
        (shape["x"] - 30, shape["y"] - 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 100),
        2,
    )


cv2.putText(
    image,
    f"<= Answer",
    (result["x"] + 10, result["y"]),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2,
)

cv2.imshow("Hollow Shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
