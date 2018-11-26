import argparse
import os
import skimage.io
import torch
import fnmatch
import re
import isic
import model as modellib
import visualize as visualize

ROOT_DIR = os.getcwd()

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
ISIC_MODEL_PATH = os.path.join(LOGS_DIR, "2017_200_epochs_with_mean", "mask_rcnn_isic_0180.pth")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

IGNORE_VAL = "ISIC_0013945, ISIC_0006815, ISIC_0013863"
IGNORE_TEST = "ISIC_0012147, ISIC_0012941"
IGNORE = IGNORE_VAL + ", " + IGNORE_TEST


class InferenceConfig(isic.ISICConfig):
    IMAGES_PER_GPU = 1


def filter_by_file_types(root, files, file_types):
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def predict(images_dir):
    config = InferenceConfig()
    config.display()

    # Create model object.
    model = modellib.MaskRCNN(model_dir=LOGS_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()

    # Load weights trained
    model.load_state_dict(torch.load(ISIC_MODEL_PATH))

    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    images = filter_by_file_types(images_dir, os.listdir(images_dir), ["*.jpeg", "*.jpg"])
    images = sorted(images)
    total_images = len(images)
    cont = 0

    class_names = ["BG", "Lesion"]

    for image in images:
        image_name = image.split("/")[-1][:-4]

        if image_name not in IGNORE:
            img = skimage.io.imread(image)

            result = model.detect([img])
            pred = result[0]

            output_name = os.path.join(OUTPUTS_DIR, image_name + ".png")
            #visualize.display_instances(
                #img, pred["rois"], pred["masks"], title=output_name)
            visualize.display_instances(img, pred['rois'], pred['masks'], pred['class_ids'],
                            class_names, output_name, pred['scores'])

        cont = cont + 1
        print("Processed {}/{} images.".format(cont, total_images))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--images-dir", required=True,
                        help="Dir where the images are located.")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    predict(args.images_dir)