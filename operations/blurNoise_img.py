import cv2
import pybboxes as pbx

image_loc = '/home/s-suryan/Documents/Projects/Python/AugmentiX/temp/build/License-Plate-Recognition-4/test/images/00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.jpg'
label_loc = '/home/s-suryan/Documents/Projects/Python/AugmentiX/temp/build/License-Plate-Recognition-4/test/labels/00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.txt'
with open(label_loc) as f:
    lines = f.read()
    _, x, y, w, h = map(float, lines.split(' '))
    class_id = int(_)
    yolo_bbox = (x, y, w, h)

    image = cv2.imread(image_loc)
    image_copy = image.copy()
    H, W = image.shape[:2]
    coordinates = pbx.convert_bbox(yolo_bbox, from_type="yolo", to_type="coco", image_size=(W, H))
    cv2.rectangle(image, coordinates, (0, 0, 255), 2)
    cv2.imshow('Image with Rectangle', image)

    blurred = cv2.blur(image, (10, 10), borderType=cv2.BORDER_DEFAULT)

    cv2.rectangle(blurred, coordinates, (0, 0, 255), 2)
    cv2.imshow('Image flipped', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()