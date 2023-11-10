import os
import uuid

import cv2
import numpy as np
import pybboxes as pbx


class Augmentor:
    """
    This is the class holding all the augmentation methods
    """

    def __init__(self):
        pass

    @staticmethod
    def flip_img(img, fObj, flip_codes) -> bool:
        """
        This function flips the image and updates bounding box coordinates and labels accordingly.
        :param img: Input image.
        :param fObj: File object
        :param flip_codes: Flip code (0 for horizontal, 1 for vertical, -1 for both).
        :return: True if successful, False if an error occurs.
        """
        try:
            lines = fObj.read().split('\n')
            for flip_code in flip_codes:
                code = str(uuid.uuid4().hex)
                label = []
                image_path = os.path.join('temp', 'output', 'images', code + '.jpg')
                label_path = os.path.join('temp', 'output', 'labels', code + '.txt')
                H, W = img.shape[:2]
                flipped = cv2.flip(img, flipCode=flip_code)
                for line in lines:
                    line = line.split(' ')
                    if line[0] == '':
                        with open(label_path, 'x') as f:
                            f.write('')
                        cv2.imwrite(image_path, flipped)
                        return True
                    else:
                        class_id = int(line[0])
                        x, y, w, h = np.array(list(map(float, line[1:])))
                        bbox = (x, y, w, h)

                        # Update bounding box coordinates
                        x, y, w, h = pbx.convert_bbox(bbox, from_type="yolo", to_type="coco", image_size=(W, H))
                        if flip_code == 0:  # horizontal flip
                            y = H - y - h
                        elif flip_code == 1:  # vertical flip
                            x = W - x - w
                        elif flip_code == -1:  # horizontal and vertical flip
                            x = W - x - w
                            y = H - y - h

                        # Convert back to YOLO format
                        yolo_bbox = pbx.convert_bbox((x, y, w, h), from_type="coco", to_type="yolo", image_size=(W, H))
                        label.append(f"{class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}")

                cv2.imwrite(image_path, flipped)
                with open(label_path, 'x') as f:
                    labels = '\n'.join(label)
                    f.write(labels.rstrip())
                return True
        except Exception as e:
            print(e)
            return False

    @staticmethod
    def noise_blur_img(img, fObj, blur_codes) -> bool:
        """
        This function adds noise and blur to the image and updates bounding box coordinates and labels accordingly.
        :param img: Input image.
        :param fObj: File object
        :param blur_codes: Blur code contains the extra args for the blur function.
        :return: True if successful, False if an error occurs.
        """
        try:
            lines = fObj.read().split('\n')
            for blur_code in blur_codes:
                code = str(uuid.uuid4().hex)
                label = []
                image_path = os.path.join('temp', 'output', 'images', code + '.jpg')
                label_path = os.path.join('temp', 'output', 'labels', code + '.txt')
                kernel_size, border_type = blur_code

                blurred = cv2.blur(img, kernel_size, borderType=border_type)

                for line in lines:
                    line = line.split(' ')
                    if line[0] == '':
                        with open(label_path, 'x') as f:
                            f.write('')
                        cv2.imwrite(image_path, blurred)
                        return True
                    else:
                        class_id = int(line[0])
                        x, y, w, h = np.array(list(map(float, line[1:])))
                        bbox = (x, y, w, h)
                        # Update bounding box coordinates
                        label.append(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}")

                cv2.imwrite(image_path, blurred)
                with open(label_path, 'x') as f:
                    labels = '\n'.join(label)
                    f.write(labels.rstrip())
                return True

        except Exception as e:
            print(e)
            return False