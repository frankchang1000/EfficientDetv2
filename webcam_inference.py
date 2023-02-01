# run model on live webcam feed

import cv2
import os
import numpy as np
import tensorflow as tf
from utils.postprocess import FilterDetections
from utils.visualize import draw_boxes
from utils.file_reader import parse_label_file
from typing import Union

# Load label file
labels_dict = parse_label_file("data/datasets/v2/labels.txt")


def preprocess_image(image_path: str, 
                     image_dims: tuple) -> Union[tf.Tensor, tuple]:
    """Preprocesses an image.
        
    Parameters:
        image_path: Path to the image
        image_dims: The dimensions to resize the image to
    Returns:
        A preprocessed image with range [0, 255]
        A Tuple of the original image shape (w, h)
    """
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image)
    original_shape = tf.shape(image)
    image = tf.image.resize(images=image,
                            size=image_dims,
                            method="bilinear")
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, tf.float32)
    # Image is on scale [0-255]
    return image, (original_shape[1], original_shape[0])

def test(model: tf.keras.models.Model,
         image_dims: tuple, 
         label_dict: dict, 
         score_threshold: float, 
         iou_threshold: float) -> None:
    counter = 0
    while True:
        counter += 1
        frame, image_np = cap.read(0)
        # save image frame to disk
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"D:/test-{counter}.jpeg", image_np)
        image_path = f"D:/test-{counter}.jpeg"

        image, original_shape = preprocess_image(image_path, image_dims)

        pred_cls, pred_box = model(image, training=False)
        labels, bboxes, scores = FilterDetections(
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            image_dims=image_dims)(
                labels=pred_cls,
                bboxes=pred_box)

        labels = [list(label_dict.keys())[int(l)]
                for l in labels[0]]
        bboxes = bboxes[0]
        scores = scores[0]

        out_image = draw_boxes(
            image=tf.squeeze(image, axis=0),
            original_shape=original_shape,
            resized_shape=image_dims,
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            labels_dict=label_dict)

        # change pil to numpy
        out_image = np.array(out_image)
        out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
        # image = np.array(image)
        cv2.imshow("image", out_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    # Set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
    model = tf.keras.models.load_model("D:\coding\EfficientDet/v2/v2")

    test(
        model=model,
        image_dims=(512, 512),
        label_dict=labels_dict, 
        score_threshold=0.9, 
        iou_threshold=0.7)
