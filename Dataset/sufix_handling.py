import os
import argparse
from tqdm import tqdm

from filters import filter_for_images


def check_inputs(images_dir, sufix, replace):
    if not os.path.exists(images_dir):
        print("\nThe given images directory \"{}\" does not exists, check "
            "again and try it later.\n".format(images_dir))
        return None, None

    images = filter_for_images(images_dir, os.listdir(images_dir))
    images_quantity = len(images)

    if images_quantity == 0:
        print("\nNo images were found in \"{}\" directory. Skipping sufix "
            "renaming...\n".format(images_dir))
        return None, None

    if replace is None:
        answer = input("\nFound {} images, proceding to add \"{}\" to the end "
            "of the images' name. Continue? (y/n): "
            .format(images_quantity, sufix))
    else:
        answer = input("\nFound {} images, proceding to replace \"{}\" with "
            "\"{}\" in the images' name. Continue? (y/n): "
            .format(images_quantity, replace, sufix))

    if not str(answer).lower() == "y":
        answer = input("\nAre you sure you want to cancel? (y/n): ")

        if str(answer).lower() == "y":
            return None, None

    print("\n")
    return images, images_quantity


def change_images_sufix_in_folder(images_dir, sufix, replace, sep):
    images, images_quantity = check_inputs(images_dir, sufix, replace)

    if images is None or images_quantity is None:
        print("\n")
        return

    pbar = tqdm(desc="Adding sufix", total=images_quantity)

    for file_name in images:
        if replace is not None:
            # This is to avoid replacing posible similar words in the image
            # path, and restricting it to only the image name
            new_name = file_name.split("/")
            name_only = new_name.pop()
            name_only = name_only.replace(replace, sufix)
            new_name = "/".join(new_name)
            new_name = os.path.join(new_name, name_only)
        else:
            extension = file_name[-4:]
            new_name = file_name[:-4]
            new_name = "{}{}{}{}".format(new_name, sep, sufix, extension)

        # Save the image with its new name
        os.rename(file_name, new_name)
        pbar.update(1)
    
    pbar.close()
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--images-dir", help="The directory where the images "
        "you want to rename are located.", required=True)
    parser.add_argument("--sufix", help="The new sufix that is going to be "
        "added to the images.", required=True)
    parser.add_argument("--separator", help="The separator of the words in "
        "image name. It will be ignored if a replace word is provided.", 
        default=" ")
    parser.add_argument("--replace", help="The word you want to replace "
        "with the new sufix", default=None)

    args = parser.parse_args()

    change_images_sufix_in_folder(args.images_dir, args.sufix, 
        args.replace,  args.separator)