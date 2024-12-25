import cv2
import numpy as np


def get_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    else:
        return 0, 0


def get_shapes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (13, 13), 0)
    edges = cv2.Canny(blurred, 5, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 选择合适的膨胀核
    dilated = cv2.dilate(closed_edges, kernel2, iterations=1)

    contours, hierarchy = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    outer_shapes = []
    inner_shapes = []

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < 900 or area > 100000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
        vertices = len(approx)

        if hierarchy[0][i][2] != -1:
            outer_shapes.append(
                {
                    "vertices": vertices,
                    "index": i,
                    "contour": contour,
                }
            )
        else:
            cx, cy = get_center(contour)
            inner_shapes.append(
                {
                    "vertices": vertices,
                    "contour": contour,
                    "x": cx,
                    "y": cy,
                    "parent_index": hierarchy[0][i][3],
                }
            )

    result = get_filtered_shapes(outer_shapes, inner_shapes)
    return result, image


def get_filtered_shapes(outer_shapes, inner_shapes):
    parent_dict = {parent["index"]: parent for parent in outer_shapes}
    filtered_child_array = []
    for child in inner_shapes:
        parent_index = child["parent_index"]
        if parent_index in parent_dict:
            parent = parent_dict[parent_index]
            new_child = {
                **child,
                "parent_vertices": parent["vertices"],
                "parent_contour": parent["contour"],
            }
            filtered_child_array.append(new_child)
    return filtered_child_array
