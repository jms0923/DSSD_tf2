import argparse
import tensorflow as tf
import os
import sys
import time
import yaml

from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from voc_data import create_batch_generator
from anchor import generate_default_boxes
from network import create_ssd
from resnet101_network import create_dssd
from losses import create_losses


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    default="/home/ubuntu/minseok/dataset/LG_Classfication/LSK_VOC/",
    type=str,
)
parser.add_argument("--data-year", default="2007")
parser.add_argument("--arch", default="dssd512")   # ssd300
parser.add_argument("--batch-size", default=24, type=int)
parser.add_argument("--num-batches", default=-1, type=int)
parser.add_argument("--neg-ratio", default=3, type=int)
parser.add_argument("--initial-lr", default=1e-3, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--weight-decay", default=5e-4, type=float)
parser.add_argument("--num-epochs", default=4000, type=int)
parser.add_argument("--checkpoint-dir", default="/home/ubuntu/minseok/DSSD_tf2/checkpoints/dssd512")
parser.add_argument("--checkpoint-path", default="/home/ubuntu/minseok/DSSD_tf2/checkpoints/network120/network1200")
parser.add_argument("--pretrained-type", default="base")    # specified
parser.add_argument("--gpu-id", default="0")

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

NUM_CLASSES = 8


@tf.function
def train_step(imgs, gt_confs, gt_locs, network, criterion, optimizer):
    with tf.GradientTape() as tape:
        confs, locs = network(imgs)
        conf_loss, loc_loss = criterion(confs, locs, gt_confs, gt_locs)
        loss = conf_loss + loc_loss
        l2_loss = [tf.nn.l2_loss(t) for t in network.trainable_variables]
        l2_loss = args.weight_decay * tf.math.reduce_sum(l2_loss)
        loss += l2_loss
    gradients = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))

    return loss, conf_loss, loc_loss, l2_loss


if __name__ == "__main__":
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    with open("./config.yml") as f:
        cfg = yaml.load(f)

    try:
        config = cfg[args.arch.upper()]
    except AttributeError:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    default_boxes = generate_default_boxes(config)

    print("args.data_dir : ", args.data_dir)

    batch_generator, val_generator, info = create_batch_generator(
        args.data_dir,
        args.data_year,
        default_boxes,
        config["image_size"],
        args.batch_size,
        args.num_batches,
        mode="train",
        augmentation=["flip"],
    )  # the patching algorithm is currently causing bottleneck sometimes
    print('batch_generator info : ', info)

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
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_path=args.checkpoint_path,
                config=config
            )
        
    except Exception as e:
        print(e)
        print("The program is exiting...")
        sys.exit()

    criterion = create_losses(args.neg_ratio, NUM_CLASSES)

    steps_per_epoch = info["length"] // args.batch_size

    lr_fn = PiecewiseConstantDecay(
        boundaries=[
            int(steps_per_epoch * args.num_epochs * 2 / 3),
            int(steps_per_epoch * args.num_epochs * 5 / 6),
        ],
        values=[args.initial_lr, args.initial_lr * 0.1, args.initial_lr * 0.01],
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn, momentum=args.momentum)

    train_log_dir = "logs/train"
    val_log_dir = "logs/val"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        avg_conf_loss = 0.0
        avg_loc_loss = 0.0
        start = time.time()
        for i, (_, imgs, gt_confs, gt_locs) in enumerate(batch_generator):
            loss, conf_loss, loc_loss, l2_loss = train_step(
                imgs, gt_confs, gt_locs, network, criterion, optimizer
            )
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)
            avg_conf_loss = (avg_conf_loss * i + conf_loss.numpy()) / (i + 1)
            avg_loc_loss = (avg_loc_loss * i + loc_loss.numpy()) / (i + 1)
            # if (i + 1) % 50 == 0:
        print(
            "Epoch: {} Batch {} Time: {:.2}s | Loss: {:.4f} Conf: {:.4f} Loc: {:.4f}".format(
                epoch + 1,
                i + 1,
                time.time() - start,
                avg_loss,
                avg_conf_loss,
                avg_loc_loss,
            )
        )

        avg_val_loss = 0.0
        avg_val_conf_loss = 0.0
        avg_val_loc_loss = 0.0

        for i, (_, imgs, gt_confs, gt_locs) in enumerate(val_generator):
            val_confs, val_locs = network(imgs)
            val_conf_loss, val_loc_loss = criterion(
                val_confs, val_locs, gt_confs, gt_locs
            )
            val_loss = val_conf_loss + val_loc_loss
            avg_val_loss = (avg_val_loss * i + val_loss.numpy()) / (i + 1)
            avg_val_conf_loss = (avg_val_conf_loss * i + val_conf_loss.numpy()) / (
                i + 1
            )
            avg_val_loc_loss = (avg_val_loc_loss * i + val_loc_loss.numpy()) / (i + 1)

        with train_summary_writer.as_default():
            tf.summary.scalar("loss", avg_loss, step=epoch)
            tf.summary.scalar("conf_loss", avg_conf_loss, step=epoch)
            tf.summary.scalar("loc_loss", avg_loc_loss, step=epoch)

        with val_summary_writer.as_default():
            tf.summary.scalar("loss", avg_val_loss, step=epoch)
            tf.summary.scalar("conf_loss", avg_val_conf_loss, step=epoch)
            tf.summary.scalar("loc_loss", avg_val_loc_loss, step=epoch)

        if (epoch + 1) % 50 == 0:
            network.save_weights(os.path.join(args.checkpoint_dir, "network{}".format(epoch + 1)))  # .h5
            # network.save(os.path.join(args.checkpoint_dir, str(epoch+1))) # , save_format='tf'

