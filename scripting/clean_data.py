
import os
import glob
import shutil
import random

# for each folder in images_labelled
for folder in os.listdir('D:/coding/EfficientDet/data/datasets/isef/images_labelled'):
    # for each file in the folder
    for file in os.listdir('D:/coding/EfficientDet/data/datasets/isef/images_labelled/' + folder):
        # if the jpg does not have a corresponding xml (the starting numbers are the same)
        if not os.path.exists('D:/coding/EfficientDet/data/datasets/isef/annotations/' + folder + '/' + file[:-4] + '.xml'):
            # delete the jpg
            os.remove('D:/coding/EfficientDet/data/datasets/isef/images_labelled/' + folder + '/' + file)
            print('deleted ' + file)


