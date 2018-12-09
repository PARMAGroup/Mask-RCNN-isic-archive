import os
import shutil
import argparse
from tqdm import tqdm

from filters import filter_for_jpeg, filter_for_png


def dirs_exists(dirs: dict):
    not_existing_dirs = list()

    for key in dirs.keys():
        if not os.path.exists(dirs[key]):
            not_existing_dirs.append(dirs[key])

    lenght = len(not_existing_dirs)

    if lenght > 0:
        print("\nThe dir{1} \"{0}\" could not be found, check if the path{1} {2} correct "
              "and try again later.\n".format(", ".join(not_existing_dirs),
                                            "s" if lenght > 1 else "",
                                            "are" if lenght > 1 else "is"))
        return False
    return True


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_subdataset_dirs(dirs: dict):
    for key in dirs.keys():
        create_dir(dirs[key])


def separate_dataset(isic_root_dir, train_per):
    isic_dirs = {
        "root": isic_root_dir,
        "img": os.path.join(isic_root_dir, "Images"),
        "seg": os.path.join(isic_root_dir, "Segmentation")
    }

    if not dirs_exists(isic_dirs):
        exit(1)

    images = filter_for_jpeg(isic_dirs["img"], os.listdir(isic_dirs["img"]))
    seg = filter_for_png(isic_dirs["seg"], os.listdir(isic_dirs["seg"]))

    images = sorted(images)
    seg = sorted(seg)

    imgs_quantity = len(images)
    segs_quantity = len(seg)

    if imgs_quantity != segs_quantity:
        print("\nA segmentation for each image was not found, are you missing something? "
              "{} images found and {} segmentation found.\n".format(imgs_quantity, segs_quantity))
        exit(1)

    train_images_quantity = round(imgs_quantity * train_per)
    val_images_quantity = imgs_quantity - train_images_quantity

    try:
        num_of_datasets = round(
            imgs_quantity / (imgs_quantity - train_images_quantity))
    except:
        num_of_datasets = 1

    print("\nThe dataset will be split into training and validation sets, creating "
          "{} datasets having each {} training images and {} validation images."
          .format(num_of_datasets, train_images_quantity, val_images_quantity))

    answer = str(input("\nIs the configuration correct? (y/n): "))

    if answer.lower() != "y":
        answer = str(
            input("\nThe operation will be aborted, continue? (y/n): "))

        if answer.lower() == "y":
            print("\nExiting without processing dataset.\n")
            exit(0)

    offset = 0
    pbar = tqdm(desc="Processing dataset", total=num_of_datasets * imgs_quantity)

    for dataset_id in range(num_of_datasets):
        subdataset_dir = os.path.join(
            isic_root_dir, "Dataset_{}".format(dataset_id))

        subdataset_dirs = {
            "root": subdataset_dir,
            "train_img": os.path.join(subdataset_dir, "Train/Images"),
            "train_seg": os.path.join(subdataset_dir, "Train/Segmentation"),
            "val_img": os.path.join(subdataset_dir, "Val/Images"),
            "val_seg": os.path.join(subdataset_dir, "Val/Segmentation")
        }

        if os.path.exists(subdataset_dir):
            offset += train_images_quantity
            continue

        create_subdataset_dirs(subdataset_dirs)
        train_count = 0

        for img_index in range(offset, imgs_quantity + offset):
            image_with_path = images[img_index % imgs_quantity]
            image_name = image_with_path.split("/")[-1][:-4]
            seg_name = [s for s in seg if image_name in s]

            if train_count < train_images_quantity:
                train_count += 1
                dest_dir = "train"
            else:
                dest_dir = "val"

            seg_dir_key = "{}_{}".format(dest_dir, "seg")
            shutil.copy2(seg_name[0], subdataset_dirs[seg_dir_key])

            img_dir_key = "{}_{}".format(dest_dir, "img")
            shutil.copy2(image_with_path, subdataset_dirs[img_dir_key])

            pbar.update(1)

        offset += train_images_quantity
    
    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--isic-root", help="The path to the root folder of "
                        "the isic archive ", required=True)
    parser.add_argument("--train-percentage", help="The percentage of the whole "
                        "dataset that is going to be used as training ]0.0, "
                        "1.0[.", type=float, default=0.8)

    args = parser.parse_args()

    separate_dataset(args.isic_root, args.train_percentage)
