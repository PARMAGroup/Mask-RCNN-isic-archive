import os
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
from PIL import Image
import utils

plt.ioff()

def apply_mask(image, mask, color):
    """Apply the given mask to the image.
    """
    image = np.where(mask == 1,
                                  255,
                                  0)
    return image


def display_instances(image, boxes, masks, title="",
                      figsize=(16, 16), ax=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    figsize: (optional) the size of the image.
    """
    # Number of instances
    N = boxes.shape[0]
    
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1]

    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)

    # Generate random colors
    color = (1., 1., 1.)
    ax.axis('off')

    masked_image = np.zeros(image.shape[:2], dtype=np.uint32)
    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        # Mask
        mask = masks[:, :, i]
        masked_image = np.maximum(masked_image, apply_mask(masked_image, mask, color))

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)

        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    ax.imshow(masked_image.astype(np.uint8))
    final_image = Image.fromarray(masked_image.astype(np.uint8))
    final_image.save(title)
    plt.close()