from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet101
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import os

from layers import create_vgg16_layers, create_extra_layers, create_conf_head_layers, create_loc_head_layers
from resnet101_layers import create_resnet101_layers, create_ssd_layers, create_deconv_layer, create_prediction_layer, create_cls_head_layers, create_loc_head_layers


class DSSD(Model):
    """ Class for SSD model
    Attributes:
        num_classes: number of classes
    """

    def __init__(self, num_classes, arch='dssd320', config=None):
        super(DSSD, self).__init__()
        self.num_classes = num_classes
        self.resnet101_conv3, self.resnet101_conv5 = create_resnet101_layers()

        # print(self.resnet101_conv3.summary())
        # print(self.resnet101_conv5.summary())

        self.batch_norm = layers.BatchNormalization(
            beta_initializer='glorot_uniform',
            gamma_initializer='glorot_uniform'
        )
        self.ssd_layers = create_ssd_layers()

        self.deconv_layers = []
        for idx, resol in enumerate(config['deconv_resolutions']):
            # print("config['fm_sizes'] : ", config['fm_sizes'][idx], config['fm_sizes'][idx+1])
            self.deconv_layers.append(create_deconv_layer(idx, (config['fm_sizes'][idx], config['fm_sizes'][idx+1]), resol))

        self.prediction_modules = []
        for idx in range(6):
            self.prediction_modules.append(create_prediction_layer(idx, self.num_classes))
        
        self.cls_head_layers = create_cls_head_layers(num_classes)
        self.loc_head_layers = create_loc_head_layers()

        # self.init_resnet101()

        # if arch == 'ssd300':
        #     self.extra_layers.pop(-1)
        #     self.conf_head_layers.pop(-2)
        #     self.loc_head_layers.pop(-2)


    # def compute_heads(self, x, idx):
    def compute_heads(self, conf, loc):
        """ Compute outputs of classification and regression heads
        Args:
            x: the input feature map
            idx: index of the head layer
        Returns:
            conf: output of the idx-th classification head
            loc: output of the idx-th regression head
        """
        # conf = self.cls_head_layers[idx](x)
        conf = tf.reshape(conf, [conf.shape[0], -1, self.num_classes])

        # loc = self.loc_head_layers[idx](x)
        loc = tf.reshape(loc, [loc.shape[0], -1, 4])

        return conf, loc


    def call(self, x):
        """ The forward pass
        Args:
            x: the input image
        Returns:
            confs: list of outputs of all classification heads
            locs: list of outputs of all regression heads
        """
        confs = []
        locs = []
        head_idx = 0
        features = []

        x = self.resnet101_conv3(x)

        conv3_feature = x
        features.append(conv3_feature)

        x = self.resnet101_conv5(x)
        conv5_feature = x

        for layer in self.ssd_layers:
            x = layer(x)
            features.append(x)
        # print('x before compute_heads : ', x.get_shape().as_list())

        # block 10 (last ssd layer)
        conf, loc = self.prediction_modules[0](features.pop(-1))
        conf, loc = self.compute_heads(conf, loc)
        # conf, loc = self.compute_heads(pred, 0)
        confs.append(conf)
        locs.append(loc)

        for order in range(len(features)):
            # print('\nfeature call num : ', order)
            # print('x shape : ', x.get_shape().as_list())
            # print('feature shape : ', features[-1].get_shape().as_list(), '\n')
            x = self.deconv_layers[order]([x, features.pop(-1)])
            conf, loc = self.prediction_modules[order+1](x)
            conf, loc = self.compute_heads(conf, loc)
            # conf, loc = self.compute_heads(pred, order+1)
            # print('order : ', order, conf)
            confs.append(conf)
            locs.append(loc)

        confs = tf.concat(confs, axis=1)
        locs = tf.concat(locs, axis=1)

        # print('net confs return : ', confs.shape, confs)
        # print('net locs return : ', locs.shape, locs)

        return confs, locs


def create_dssd(num_classes, arch, pretrained_type,
               checkpoint_dir=None,
               checkpoint_path=None,
               config=None):
    """ Create SSD model and load pretrained weights
    Args:
        num_classes: number of classes
        pretrained_type: type of pretrained weights, can be either 'VGG16' or 'ssd'
        weight_path: path to pretrained weights
    Returns:
        net: the SSD model
    """
    net = DSSD(num_classes, arch, config)
    # net.load_weights('/home/ubuntu/minseok/DSSD_tf2/checkpoints/network4')
    if pretrained_type == 'base':
        # he layer initiate when declare layers
        pass

    elif pretrained_type == 'latest':
        try:
            paths = [os.path.join(checkpoint_dir, path)
                     for path in os.listdir(checkpoint_dir)]
            latest = sorted(paths, key=os.path.getmtime)[-1]
            net.load_weights(latest)
        except AttributeError as e:
            print('Please make sure there is at least one checkpoint at {}'.format(
                checkpoint_dir))
            print('The model will be loaded from base weights.')
            # net.init_vgg16()
        except ValueError as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    latest, arch))
        except Exception as e:
            print(e)
            raise ValueError('Please check if checkpoint_dir is specified')

    elif pretrained_type == 'specified':
        # print('checkpoint_path : ', checkpoint_path)

        # if not os.path.isfile(checkpoint_path):
        #     raise ValueError(
        #         'Not a valid checkpoint file: {}'.format(checkpoint_path))
        try:
            net.load_weights(checkpoint_path)
        except Exception as e:
            raise ValueError(
                'Please check the following\n1./ Is the path correct: {}?\n2./ Is the model architecture correct: {}?'.format(
                    checkpoint_path, arch))

    else:
        raise ValueError('Unknown pretrained type: {}'.format(pretrained_type))
    return net

