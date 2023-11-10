import cv2
import pybboxes as pbx
import numpy as np


def rotate_point(point, center, angle):
    x, y = point
    cx, cy = center
    rad_angle = np.radians(angle)
    x_rot = (x - cx) * np.cos(rad_angle) - (y - cy) * np.sin(rad_angle) + cx
    y_rot = (x - cx) * np.sin(rad_angle) + (y - cy) * np.cos(rad_angle) + cy
    return x_rot, y_rot


def rotate_bbox(bbox, rotation_matrix, angle, image_size):
    x, y, w, h = bbox
    bbox_points = np.array(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    bb_rotated = np.vstack((bbox_points.T, np.array((1, 1, 1, 1))))
    bb_rotated = np.dot(rotation_matrix, bb_rotated).T
    rotated_bbox = (
        np.append(bb_rotated[:, 0], bb_rotated[0, 0]),
        np.append(bb_rotated[:, 1], bb_rotated[0, 1])
        )
    return rotated_bbox


image_loc = '/home/s-suryan/Documents/Projects/Python/AugmentiX/temp/build/License-Plate-Recognition-4/test/images/00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.jpg'
label_loc = '/home/s-suryan/Documents/Projects/Python/AugmentiX/temp/build/License-Plate-Recognition-4/test/labels/00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.txt'

with open(label_loc) as f:
    lines = f.read()
    _, x, y, w, h = map(float, lines.split(' '))
    class_id = int(_)
    yolo_bbox = (x, y, w, h)

    image = cv2.imread(image_loc)
    image_copy = image.copy()
    angle = 45
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    coordinates = pbx.convert_bbox(yolo_bbox, from_type="yolo", to_type="coco", image_size=(width, height))
    print(coordinates)
    cv2.rectangle(image, coordinates, (0, 0, 255), 2)
    cv2.imshow('Image with Rectangle', image)

    rotated_bbox = rotate_bbox(coordinates, rotation_matrix, 45, (width, height))
    # rotated_bbox = map(int, rotated_bbox)
    # rotated_bbox = tuple(rotated_bbox)
    print(rotated_bbox)

    # cv2.rectangle(rotated_image, rotated_bbox, (0, 0, 255), 2)
    # cv2.imshow('Image rotated', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()