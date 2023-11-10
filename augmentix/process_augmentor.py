import cv2


class ProcessAugmentor:
    """
    Augmentor that processes data with a given function.
    """

    def __init__(self):
        pass

    @staticmethod
    def process_image(args, success, failed):
        """
        This function processes the operation type and params and redirects it to correspondingly.
        :param args: Arguments to be passed to the child functions.
        :param success: Success counter.
        :param failed: Failed counter.
        :return:
        """
        aug, params, image_path, label_path = args
        # operations, fields = params
        # print(fields)
        for operation, field in params.items():
            if operation == 'flip_img':
                with open(label_path) as fobj:
                    status = aug.flip_img(cv2.imread(image_path), fobj, field['flip_code'])
                    if status:
                        success.value += 1
                    else:
                        failed.value += 1
            elif operation == 'noise_blur_img':
                with open(label_path) as fobj:
                    status = aug.noise_blur_img(cv2.imread(image_path), fobj, field['noise_type'])
                    if status:
                        success.value += 1
                    else:
                        failed.value += 1
            else:
                raise Exception('Invalid operation type')