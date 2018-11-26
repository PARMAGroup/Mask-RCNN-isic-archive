from config import Config
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

import numpy as np
import utils
import math
import time
import os

import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()
ISIC_TRAIN_DIR = "../../ISIC_Challenge_2017/Training/"
ISIC_VAL_DIR = "../../ISIC_Challenge_2017/Val/"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class ISICConfig(Config):
    NAME = "ISIC"

    GPU_COUNT = 1

    IMAGES_PER_GPU = 8

    NUM_CLASSES = 2  # Background and lesion

    VALIDATION_STEPS = 50

    MEAN_PIXEL = np.array([180.65, 148.44, 136.69])


class ISICDataset(utils.Dataset):
    def load_isic(self, dataset_dir, subset):
        isic = COCO("{0}/annotations_isic_{1}.json".format(dataset_dir, subset))
        image_dir = "{}/Images".format(dataset_dir)

        class_ids = sorted(isic.getCatIds())
        image_ids = list(isic.imgs.keys())

        for i in class_ids:
            self.add_class("isic", i, isic.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image("isic", image_id=i, path=os.path.join(image_dir, isic.imgs[i]["file_name"]), width=isic.imgs[i]["width"],
                           height=isic.imgs[i]["height"],
                           annotations=isic.loadAnns(isic.getAnnIds(
                               imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]

        for ann in annotations:
            class_id = self.map_source_class_id("isic.{}".format(ann['category_id']))

            if class_id:
                m = self.annToMask(ann, image_info["height"], image_info["width"])

                if m.max() < 1:
                    continue

                if ann['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)

                instance_masks.append(m)
                class_ids.append(class_id)
        
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(ISICDataset, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


if __name__ == "__main__":
    config = ISICConfig()
    config.display()

    model = modellib.MaskRCNN(config=config, model_dir=DEFAULT_LOGS_DIR)

    if config.GPU_COUNT:
        model = model.cuda()

    model.load_weights(config.IMAGENET_MODEL_PATH)

    dataset_train = ISICDataset()
    dataset_train.load_isic(ISIC_TRAIN_DIR, "Train")
    dataset_train.prepare()

    dataset_val = ISICDataset()
    dataset_val.load_isic(ISIC_VAL_DIR, "Val")
    dataset_val.prepare()

    start_time = time.time()

    # Training - Stage 1
    print("Training network heads")
    model.train_model(dataset_train, dataset_val,
                learning_rate=0.01,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train_model(dataset_train, dataset_val,
                learning_rate=0.01,
                epochs=120,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train_model(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=200,
                layers='all')

    elapsedTime = time.time() - start_time
    hours = math.floor(elapsedTime / (60*60))
    elapsedTime = elapsedTime - hours * (60*60)
    minutes = math.floor(elapsedTime / 60)
    elapsedTime = elapsedTime - minutes * (60)
    seconds = math.floor(elapsedTime)
    elapsedTime = elapsedTime - seconds
    ms = elapsedTime * 1000
    if(hours != 0):
        message = "%d hours %d minutes %d seconds" % (hours, minutes, seconds)
    elif(minutes != 0):
        message = "%d minutes %d seconds" % (minutes, seconds)
    else :
        message = "%d seconds %f ms" % (seconds, ms)

    print("\n\nTraining took {} to complete.".format(message))
