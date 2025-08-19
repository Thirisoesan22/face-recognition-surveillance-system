import cv2
import os

from augment_image import augment_image


# def create_augmented_image(path, background):
#     folder, filename = os.path.split(path)
#     res = augment_image(cv2.imread(path), background)
#     os.chdir(folder)
#     name, extension = os.path.splitext(filename)
#     new_filename = f"{name}_aug{extension}"
#     cv2.imwrite(new_filename, res)


def rewrite_to_augmented(path, background):
    folder, filename = os.path.split(path)
    res = augment_image(cv2.imread(path), background)
    os.chdir(folder)
    cv2.imwrite(filename, res)
