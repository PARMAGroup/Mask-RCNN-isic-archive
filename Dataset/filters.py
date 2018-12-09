import os
import re
import fnmatch


def filter_by_file_types(root, files, file_types):
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_jpeg(root, files):
    return filter_by_file_types(root, files, ['*.jpeg', '*.jpg'])


def filter_for_png(root, files):
    return filter_by_file_types(root, files, ['*.png'])


def filter_for_images(root, files):
    return filter_by_file_types(root, files, ['*.jpeg', '*.jpg', '*.png'])


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files
