import datetime
import json
import os
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from pycococreatortools import pycococreatortools

from filters import filter_for_jpeg, filter_for_annotations

INFO = {
    "description": "ISIC Dataset",
    "url": "https://isic-archive.com/",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "epikhdez",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = [
    {
        'id': 1,
        'name': 'lesion',
        'supercategory': 'segmentation',
    },
]


def create_annotations(root_dir, blacks_dir):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1
    move_image = False

    img_dir = os.path.join(root_dir, "Images")
    seg_dir = os.path.join(root_dir, "Segmentation")
    subset = root_dir.split("/")[-1]

    if not os.path.exists(blacks_dir):
        os.makedirs(blacks_dir)

    # filter for jpeg images
    image_files = filter_for_jpeg(img_dir, os.listdir(img_dir))
    message = "Processing images"

    pbar = tqdm(desc=message, total=len(image_files))

    # go through each image
    for image_filename in image_files:
        image_name_only = image_filename.split("/")[-1]
        image = Image.open(image_filename)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_filename), image.size)

        # filter for associated png annotations
        for root, _, files in os.walk(seg_dir):
            annotation_files = filter_for_annotations(root, files, image_filename)

            # go through each associated annotation
            for annotation_filename in annotation_files:
                annotation_name_only = annotation_filename.split("/")[-1]
                class_id = [x['id'] for x in CATEGORIES if x['name'] in annotation_filename][0]

                category_info = {'id': class_id, 'is_crowd': 'crowd' in image_filename}
                binary_mask = np.asarray(Image.open(annotation_filename)
                    .convert('1')).astype(np.uint8)

                annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

                if annotation_info is not None:
                    move_image = False
                    coco_output["annotations"].append(annotation_info)
                else:
                    move_image = True
                    os.rename(annotation_filename, os.path.join(blacks_dir, annotation_name_only))

                segmentation_id = segmentation_id + 1

        if move_image:
            os.rename(image_filename, os.path.join(blacks_dir, image_name_only))
        else:
            coco_output["images"].append(image_info)

        image_id = image_id + 1
        pbar.update(1)

    with open('{}/annotations_isic_{}.json'.format(root_dir, subset), 'w') as output_json_file:
        json.dump(coco_output, output_json_file)

    pbar.close()
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-dir", help="The path to the root folder of the dataset "
                        "Leave empty to use the default location.", required=True)

    args = parser.parse_args()
    blacks_dir = os.path.join(args.root_dir, "Blacks")

    print("\nGiven the directory \"{}\" means the program expects to find folders named "
          "as \"Images\" and \"Segmentation\" inside. Also the images that can't "
          "be processed, along with their segmentation, will be moved to \"{}\"."
          .format(args.root_dir, blacks_dir))
    answer = input("\nIs this correct? (y/n): ")
    print("\n")

    if str(answer).lower() == "y":
        create_annotations(args.root_dir, blacks_dir)
    else:
        print("\nExiting...")

    exit()
