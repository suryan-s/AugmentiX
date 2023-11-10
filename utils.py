import filecmp
import os
import shutil
import uuid

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def startup_sequence():
    """
    This function is called when the program is started.
    :return: None
    """
    dirs_to_create = [
        'temp',
        'temp/build',
        'temp/output/images',
        'temp/output/labels',
        'temp/release',
        ]

    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)


def ending_sequence():
    """
    This function is called when the program is ended.
    :return: None
    """
    if os.path.exists('temp'):
        os.system('rm -rf temp')


def check_folder_structures(folder_name: str) -> bool:
    """
    This function is called to check if the input dataset folder is structured for YOLO files
    :param folder_name:
    :return:
    """
    base_path = f'temp/build/{folder_name}'
    subfolders = ['test', 'train', 'valid']

    return all(os.path.exists(os.path.join(base_path, folder, 'images')) and
               os.path.exists(os.path.join(base_path, folder, 'labels')) for folder in subfolders)


def are_folders_identical(folder1, folder2):
    """
    Function to check if the folder1 and folder2 are identical
    :param folder1:
    :param folder2:
    :return:
    """
    # Compare the two folders recursively
    dcmp = filecmp.dircmp(folder1, folder2)

    # If any left_only or right_only entries are found, the folders are not identical
    if dcmp.left_only or dcmp.right_only:
        return False

    # Recursively check subfolders
    for common_dir in dcmp.common_dirs:
        subfolder1 = f"{folder1}/{common_dir}"
        subfolder2 = f"{folder2}/{common_dir}"

        if not are_folders_identical(subfolder1, subfolder2):
            return False

    # If no differences are found, the folders are identical
    return True


def copy_folder(src: str, dst: str) -> bool:
    """
    This function copies the selected folder to the temp folder's build directory for operations
    :param src: String (Path to the folder to be copied)
    :param dst: String (Path to the destination folder)
    :return: Bool (True if successfully copied, False if not or error)
    """

    def copy_progress(src_, dst_):
        """
        :param src_:
        :param dst_:
        """
        pbar.update()
        shutil.copy2(src_, dst_)

    src_folder = os.path.basename(src)
    dst = f'{dst}/{src_folder}'
    if not os.path.exists(src):
        print(f"Source directory '{src}' does not exist.")
        return False
    if os.path.exists(dst):
        is_exist = are_folders_identical(src, dst)
        if is_exist:
            print("Source directory already exists in build")
            return True
    else:
        os.makedirs(dst)
    try:
        total_files = sum([len(files) for root, dirs, files in os.walk(src)])
        with tqdm(total=total_files, unit="files") as pbar:
            shutil.copytree(src, dst, copy_function=copy_progress, symlinks=True, dirs_exist_ok=True)
        print("Copy completed successfully.")
        return True
    except Exception as e:
        print(f"Error copying directory: {e}")
        return False


def split_and_move_data(source_folder, destination_folder, split_ratio=(0.7, 0.2, 0.1)):
    """
    Split and move data from source_folder to train, test, and valid folders in destination_folder.

    Args:
        source_folder (str): Path to the source folder containing 'images' and 'labels' subfolders.
        destination_folder (str): Path to the destination folder where 'train', 'test', and 'valid' folders will be created.
        split_ratio (tuple): A tuple representing the split ratio for train, test, and valid sets. Default is (0.7, 0.2, 0.1).
    """
    images_folder = os.path.join(source_folder, 'images')
    labels_folder = os.path.join(source_folder, 'labels')

    # Create destination folders
    train_folder = os.path.join(destination_folder, 'train')
    test_folder = os.path.join(destination_folder, 'test')
    valid_folder = os.path.join(destination_folder, 'valid')
    source_folder_name = os.path.basename(destination_folder)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(test_folder, 'labels'), exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'labels'), exist_ok=True)

    shutil.move(src=os.path.join('temp', 'build', source_folder_name, 'data.yaml'),
                dst=os.path.join('temp', 'release', source_folder_name, 'data.yaml'))

    # Get list of files in images folder
    image_files = os.listdir(images_folder)

    # Split the data using train_test_split
    train_files, test_valid_files = train_test_split(image_files, test_size=split_ratio[1] + split_ratio[2],
                                                     random_state=42)
    test_files, valid_files = train_test_split(test_valid_files,
                                               test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]),
                                               random_state=42)

    # Move files to train, test, and valid folders
    for file in train_files:
        move_file(images_folder, labels_folder, file, train_folder)

    for file in test_files:
        move_file(images_folder, labels_folder, file, test_folder)

    for file in valid_files:
        move_file(images_folder, labels_folder, file, valid_folder)


def move_file(images_folder, labels_folder, file, destination_folder):
    source_image_path = os.path.join(images_folder, file)
    source_label_path = os.path.join(labels_folder, file.replace('.jpg', '.txt'))

    destination_image_path = os.path.join(destination_folder, 'images', file)
    destination_label_path = os.path.join(destination_folder, 'labels', file.replace('.jpg', '.txt'))

    shutil.move(source_image_path, destination_image_path)
    shutil.move(source_label_path, destination_label_path)


def move_build_files(source_folder):
    """
    This function moves the build files to the output folder
    :param source_folder:
    :return:
    """
    source_folder_name = os.path.basename(source_folder)
    dirs_to_search = [
        f'temp/build/{source_folder_name}/test',
        f'temp/build/{source_folder_name}/train',
        f'temp/build/{source_folder_name}/valid'
        ]

    for directory in dirs_to_search:
        images_folder = os.path.join(directory, 'images')
        labels_folder = os.path.join(directory, 'labels')

        for file in os.listdir(images_folder):
            image_path = os.path.join(images_folder, file)
            label_path = os.path.join(labels_folder, file.replace('.jpg', '.txt'))

            if os.path.exists(image_path) and os.path.exists(label_path):
                new_image_name = f'{str(uuid.uuid4().hex)}.jpg'
                new_label_name = new_image_name.replace('.jpg', '.txt')

                try:
                    os.rename(src=image_path, dst=os.path.join(images_folder, new_image_name))
                    os.rename(src=label_path, dst=os.path.join(labels_folder, new_label_name))

                    shutil.move(src=os.path.join(images_folder, new_image_name),
                                dst=os.path.join('temp', 'output', 'images', new_image_name))
                    shutil.move(src=os.path.join(labels_folder, new_label_name),
                                dst=os.path.join('temp', 'output', 'labels', new_label_name))
                except Exception as e:
                    print(f"Error moving files: {e}")
            else:
                print(f"Error: Either {image_path} or {label_path} does not exist.")


if __name__ == "__main__":
    # Example usage
    # source_folder = 'path/to/source/folder'
    # destination_folder = 'path/to/destination/folder'
    # split_and_move_data(source_folder, destination_folder)

    startup_sequence()
    # result = copy_folder('/home/s-suryan/Videos/License-Plate-Recognition-4', 'temp/build')
    # print ("Copy status : " + str(result))
    # result = check_folder_structures(os.path.basename('/home/s-suryan/Videos/License-Plate-Recognition-4'))
    # print("Folder check status : " + str(result))
    move_build_files('/temp/build/License-Plate-Recognition-4')

    # ending_sequence()