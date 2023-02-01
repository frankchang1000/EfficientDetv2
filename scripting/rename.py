# rename all image files in a directory to a numerical sequence and rename the corresponding xml files accordingly
# reserve the last 10% of the images for testing and rename the corresponding xml files accordingly
# add the 10% of images to test folder
# add the 10% of image numbers to val.txt
# after the rename, add the the rest of the image numbers to train.txt

import os
import glob
import shutil
import random

count = 0

# for each folder in D:/coding/EfficientDet/data/datasets/isef/images_labelled
for folder in glob.glob('D:/coding/EfficientDet/data/datasets/isef/images_labelled/*'):

    # for each file in each folder
    for file in glob.glob(folder + '/*'):
        # if the file is a jpg
        if file.endswith('.jpg'):
            # rename the jpg file
            os.rename(file, folder + '/image' + str(count) + '.jpg')
            # rename the corresponding xml file
            os.rename(file[:-4] + '.xml', folder + '/image' + str(count) + '.xml')
            count += 1
            
            # move the the image to the Images folder
            shutil.move(folder + '/image' + str(count) + '.jpg', 'D:/coding/EfficientDet/data/datasets/isef/Images')
            # write the image number to train.txt
            with open('D:/coding/EfficientDet/data/datasets/isef/train.txt', 'a') as f:
                f.write('image' + str(count) + '.jpg' + '')






