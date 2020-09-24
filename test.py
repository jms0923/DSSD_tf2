import time
import argparse
import tensorflow as tf
import os
import sys
import numpy as np
import yaml
from tqdm import tqdm

from anchor import generate_default_boxes
from box_utils import decode, compute_nms
from voc_data import create_batch_generator
from image_utils import ImageVisualizer
from losses import create_losses
from network import create_ssd
from resnet101_network import create_dssd
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="/home/ubuntu/minseok/dataset/LG_Classfication/LSK_VOC/")
parser.add_argument("--save-dir", default="outputs/images")
parser.add_argument("--data-year", default="2007")
parser.add_argument("--arch", default="dssd320")
parser.add_argument("--num-examples", default=-1, type=int)
parser.add_argument("--pretrained-type", default="specified")   # latest
parser.add_argument("--checkpoint-dir",default="/home/ubuntu/minseok/DSSD_tf2/checkpoints/")
parser.add_argument("--checkpoint-path", default="/home/ubuntu/minseok/DSSD_tf2/checkpoints/network4")
parser.add_argument("--gpu-id", default="0")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

NUM_CLASSES = 8
BATCH_SIZE = 1
THRESHOLD = 0.5

def predict(imgs, default_boxes):
    confs, locs = network(imgs)

    confs = tf.squeeze(confs, 0)
    locs = tf.squeeze(locs, 0)

    confs = tf.math.softmax(confs, axis=-1)
    classes = tf.math.argmax(confs, axis=-1)
    scores = tf.math.reduce_max(confs, axis=-1)

    boxes = decode(default_boxes, locs)

    out_boxes = []
    out_labels = []
    out_scores = []

    for c in range(1, NUM_CLASSES):
        cls_scores = confs[:, c]

        score_idx = cls_scores > THRESHOLD
        # cls_boxes = tf.boolean_mask(boxes, score_idx)
        # cls_scores = tf.boolean_mask(cls_scores, score_idx)
        cls_boxes = boxes[score_idx]
        cls_scores = cls_scores[score_idx]
        nms_idx = compute_nms(cls_boxes, cls_scores, 0.45, 200)
        cls_boxes = tf.gather(cls_boxes, nms_idx)
        cls_scores = tf.gather(cls_scores, nms_idx)
        cls_labels = [c] * cls_boxes.shape[0]

        out_boxes.append(cls_boxes)
        out_labels.extend(cls_labels)
        out_scores.append(cls_scores)

    out_boxes = tf.concat(out_boxes, axis=0)
    out_scores = tf.concat(out_scores, axis=0)

    boxes = tf.clip_by_value(out_boxes, 0.0, 1.0).numpy()
    classes = np.array(out_labels)
    scores = out_scores.numpy()

    return boxes, classes, scores


if __name__ == "__main__":
    with open("./config.yml") as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    default_boxes = generate_default_boxes(config)

    batch_generator, info = create_batch_generator(
        args.data_dir,
        args.data_year,
        default_boxes,
        config["image_size"],
        BATCH_SIZE,
        args.num_examples,
        mode="test",
    )

    try:
        if 'dssd' in args.arch:
            network = create_dssd(
                NUM_CLASSES,
                args.arch,
                args.pretrained_type,
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
        else:
            network = create_ssd(
                NUM_CLASSES,
                args.arch,
                args.pretrained_type,
                args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
    except Exception as e:
        print(e)
        print("The program is exiting...")
        sys.exit()

    os.makedirs("outputs/images", exist_ok=True)
    os.makedirs("outputs/detects", exist_ok=True)
    visualizer = ImageVisualizer(info["idx_to_name"], save_dir=args.save_dir)

    for i, (filename, imgs, gt_confs, gt_locs) in enumerate(tqdm(batch_generator, total=info["length"], desc="Testing...", unit="images")):
        start = time.time()
        boxes, classes, scores = predict(imgs, default_boxes)
        print(time.time() - start)
        filename = filename.numpy()[0].decode()
        # original_image = Image.open(os.path.join(info["image_dir"], "{}.jpg".format(filename)))
        original_image = cv2.imread(os.path.join(info["image_dir"], "{}.jpg".format(filename)), cv2.IMREAD_COLOR)

        # boxes *= original_image.size * 2
        origin_h, origin_w, _ = original_image.shape
        boxes[:,:1] *= origin_w
        boxes[:,1:2] *= origin_h
        boxes[:,2:3] *= origin_w
        boxes[:,3:4] *= origin_h
        # visualizer.save_image(original_image, boxes, classes, "{}.jpg".format(filename))
        visualizer.save_image_cv(original_image, boxes, classes, scores, "{}.jpg".format(filename))

        log_file = os.path.join("outputs/detects", "{}.txt")

        for cls, box, score in zip(classes, boxes, scores):
            cls_name = info["idx_to_name"][cls - 1]
            with open(log_file.format(cls_name), "a") as f:
                f.write(
                    "{} {} {} {} {} {}\n".format(
                        filename, score, *[coord for coord in box]
                    )
                )
