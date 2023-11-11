from __future__ import unicode_literals

import argparse
import multiprocessing
import os
import time

import cv2
import eel

from .augmentor import Augmentor
from .process_augmentor import ProcessAugmentor
from .utils import startup_sequence, split_and_move_data, move_build_files, copy_folder, ending_sequence

eel.init('web')


@eel.expose
def starter(data):
    print("Received : ", data)


@eel.expose
def run(DATASET_NAME):
    """
    Main function to run the program.
    :return:
    """
    aug = Augmentor()
    process_aug = ProcessAugmentor()
    dirs_to_search = [
        f'temp/build/{DATASET_NAME}/test',
        f'temp/build/{DATASET_NAME}/train',
        f'temp/build/{DATASET_NAME}/valid'
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


def main():
    """
    Run the program.
    """
    parser = argparse.ArgumentParser(description="AugmentiX CLI")
    parser.add_argument("--src", dest="source_folder", required=True, help="Path to the source folder")
    parser.add_argument("--dst", dest="destination_folder", required=True, help="Path to the destination folder")
    args = parser.parse_args()
    src = args.source_folder
    dst = args.destination_folder
    if not src or not dst:
        raise Exception("Please provide a valid source and destination folder")

    DATASET_NAME = src.split('/')[-1]
    print(f"Dataset name: {DATASET_NAME}")
    DATASET_DIR = src
    print(f"Dataset directory: {DATASET_DIR}")
    try:
        ending_sequence()
        startup_sequence()
        print("Startup sequence complete.")
        print("Copying build files...")
        copy_folder(DATASET_DIR, 'temp/build')
        start_time = time.time()
        print("Started main...")
        run(DATASET_NAME)
        print("Main complete.")
        print(f"Time taken for augmentation: {time.time() - start_time}")
        print("Moving build files...")
        move_build_files(DATASET_NAME)
        print("Moving build files complete.")
        print("Splitting and moving data...")
        split_and_move_data(
            'temp/output',
            f'temp/release/{DATASET_NAME}',
            zip_location=dst
            )
        print("Splitting and moving data complete.")
        if os.path.isfile(os.path.join(dst, f'{DATASET_NAME}.zip')):
            print(f"Release zip file is located at: {dst}/{DATASET_NAME}.zip")
        else:
            print("Something went wrong while checking the zip file. Please try again.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ending_sequence()