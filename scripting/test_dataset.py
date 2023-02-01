import os
import tensorflow as tf
import xml.etree.ElementTree as ET

import dataset as dataset

def find_bad(path_to_folder, images_folder="Images", labels_folder="Labels"):
    file_names = os.listdir(os.path.join(path_to_folder, images_folder))
    bad_file = False
    bad_labels = []

    for x in file_names:
        bad_file = False
        # get the base name of the file
        base_name = os.path.splitext(x)[0]
        # get the path to the xml file

        file_name = os.path.join(path_to_folder, labels_folder, base_name + ".xml")
        root = ET.parse(file_name).getroot()

        boxes = root.findall("object")
        box = []
        labels = []

        for b in boxes:
            bb = b.find("bndbox")
            if int(bb.findtext("xmin")) >= int(bb.findtext("xmax")):
                print("Bad xmin: {}".format(bb.findtext("xmin")))
                print("Bad xmax: {}".format(bb.findtext("xmax")))
                bad_file = True
            if int(bb.findtext("ymin")) >= int(bb.findtext("ymax")):
                print("Bad ymin: {}".format(bb.findtext("ymin")))
                print("Bad ymax: {}".format(bb.findtext("ymax")))
                bad_file = True
            if bad_file:
                bad_labels.append(file_name)
    return bad_labels

def remove_file(bad_labels):
    for x in bad_labels:
        print("Removing file: {}".format(x))

        #os.remove(x)

def see_dataset():
    dataset_creater = dataset.Dataset(file_names=file_names,
                                        dataset_path=args.dataset_path,
                                        labels_dict=labels_dict,
                                        batch_size=args.batch_size,
                                        shuffle_size=args.shuffle_size,
                                        images_dir=args.images_dir,
                                        labels_dir=args.labels_dir,
                                        image_dims=args.image_dims,
                                        augment_ds=args.augment_ds)

    for images, bbx, label in dataset_creater:
        for im in images:
            np.array(im)
            imshow im

        break

if __name__ == "__main__":
    # bad_labels = find_bad(path_to_folder="D:/coding/EfficientDet/data/datasets/isef")
    # remove_file(bad_labels)

