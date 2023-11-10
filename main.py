from __future__ import unicode_literals
import io
import argparse

import multiprocessing
import os
import time

import cv2
import eel

from augmentix import Augmentor, ProcessAugmentor
from utils import startup_sequence, split_and_move_data, move_build_files, copy_folder

eel.init('web')


@eel.expose
def starter(data):
    print("Received : ", data)


@eel.expose
def main():
    """
    Main function to run the program.
    :return:
    """
    aug = Augmentor()
    process_aug = ProcessAugmentor()
    dirs_to_search = [
        'temp/build/License-Plate-Recognition-4/test',
        'temp/build/License-Plate-Recognition-4/train',
        'temp/build/License-Plate-Recognition-4/valid'
        ]

    args_list = []
    operations = ['flip_img', 'noise_blur_img']
    flip_code_list = [0]
    noise_blur_list = [((10, 10), cv2.BORDER_CONSTANT)]

    for directory in dirs_to_search:
        for file in os.listdir(os.path.join(directory, 'images')):
            image_path = os.path.join(directory, 'images', file)
            label_path = os.path.join(directory, 'labels', file.replace('.jpg', '.txt'))
            params = {
                'flip_img': {
                    'flip_code': flip_code_list
                    },
                'noise_blur_img': {
                    'noise_type': noise_blur_list,
                    }
                }
            args_list.append((aug, params, image_path, label_path))

    manager = multiprocessing.Manager()
    success = manager.Value('i', 0)
    failed = manager.Value('i', 0)
    with multiprocessing.Pool() as pool:
        pool.starmap(process_aug.process_image, [(args, success, failed) for args in args_list])
    print(f"Final Status - Success: {success.value}, Failed: {failed.value}")


def run(src, dst=None):
    """
    Run the program.
    """
    DATASET_NAME = src.split('/')[-1]
    DATASET_DIR = src
    print("Starting...")
    startup_sequence()
    print("Startup sequence complete.")
    print("Copying build files...")
    copy_folder(DATASET_DIR, 'temp/build')
    print("Copying build files complete.")
    start_time = time.time()
    print("Starting main...")
    main()
    print("Main complete.")
    print(f"Time taken for augmentation: {time.time() - start_time}")
    print("Moving build files...")
    move_build_files(f'/temp/build/{DATASET_NAME}')
    print("Moving build files complete.")
    print("Splitting and moving data...")
    split_and_move_data(
        'temp/output',
        f'temp/release/{DATASET_NAME}',
        )
    print("Splitting and moving data complete.")

    print("Done.")
    print(f"Release folder is located at: temp/release/{DATASET_NAME}")


if __name__ == "__main__":
    # eel.start('index.html', size=(400, 600))
    parser = argparse.ArgumentParser(description="AugmentiX CLI")
    parser.add_argument("--src", dest="source_folder", required=True, help="Path to the source folder")
    # parser.add_argument("-d", "--dst", help="Path to the output folder")

    args = parser.parse_args()

    run(args.source_folder)