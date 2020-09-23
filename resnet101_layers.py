import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Multiply, Conv2DTranspose, Input, BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Multiply, Add
from tensorflow.keras.initializers import he_uniform


# anchors = [4, 6, 6, 6, 4, 4, 4]
anchors = [8, 8, 8, 8, 8, 8, 8]

kernel_init = he_uniform()


class layerNameCreater():

    def __init__(self, numOfCon=None, baseName=None):
        if numOfCon is not None:
            self.numOfCon = numOfCon
        else:
            self.numOfCon = 1
        self.numBlock = 1
        self.numConv = 0
        self.numDeconv = 0
        self.numBn = 0
        self.numRelu = 0
        self.numPool = 0
        if baseName is None:
            self.baseName = 'conv' + str(self.numOfCon) + '_block_' + str(self.numBlock)
        else:
            self.baseName = baseName
        # self.convName = baseName + '_conv_' + str(numConv)
        # self.bnName = baseName + '_bn_' + str(numBn)
        # self.reluName = baseName + '_relu_' + str(numRelu)


    def addBlock(self):
        self.numBlock += 1
        self.baseName = 'conv' + str(self.numOfCon) + '_block_' + str(self.numBlock)


    def call(self, kinds):
        if kinds == 'conv':
            self.numConv += 1
            return self.baseName + '_conv_' + str(self.numConv)

        elif kinds == 'deconv':
            self.numDeconv += 1
            return self.baseName + '_deconv_' + str(self.numDeconv)
            
        elif kinds == 'bn':
            self.numBn += 1
            return self.baseName + '_bn_' + str(self.numBn)

        elif kinds == 'relu':
            self.numRelu += 1
            return self.baseName + '_relu_' + str(self.numRelu)

        elif kinds == 'pool':
            self.numPool += 1
            return self.baseName + '_pool_' + str(self.numPool)

        else:
            raise Exception(kinds, 'is unknown kind of layers')


def conv1_layer(x):
    layerName = layerNameCreater(1)

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad_1')(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), name=layerName.call('conv'))(x)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = ZeroPadding2D(padding=(1,1), name='conv1_pad_2')(x)

    return x


def conv2_layer(x):
    layerName = layerNameCreater(2) 

    x = MaxPooling2D((3, 3), 2, name=layerName.call('pool'))(x)     
    shortcut = x

    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('pool'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name=layerName.call('bn'))(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            shortcut = x

        layerName.addBlock()

    return x


def conv3_layer(x):        
    layerName = layerNameCreater(3) 

    shortcut = x
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv3_bn')(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu', name=layerName.call('relu'))(x)    

            shortcut = x
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x      
        layerName.addBlock()

    return x


def conv4_layer(x):
    layerName = layerNameCreater(4)
    shortcut = x        
  
    for i in range(23):
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv4_bn')(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            shortcut = x      
        layerName.addBlock()

    return x


def conv5_layer(x):
    layerName = layerNameCreater(5)
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            shortcut = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(shortcut)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            shortcut = BatchNormalization(name='conv5_bn')(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu', name=layerName.call('relu'))(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)
            x = Activation('relu', name=layerName.call('relu'))(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid', name=layerName.call('conv'))(x)
            x = BatchNormalization(name=layerName.call('bn'))(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu', name=layerName.call('relu'))(x)       
 
            shortcut = x                  
        layerName.addBlock()
 
    return x


def create_resnet101_layers():
    input_layer = Input(shape=[None, None, 3])
    out_layer = input_layer
    out_layer = conv1_layer(out_layer)
    out_layer = conv2_layer(out_layer)
    out_layer = conv3_layer(out_layer)
    resnet101_conv3 = tf.keras.Model(input_layer, out_layer)
    # print(resnet101_conv3.summary())

    input_layer = Input(shape=[None, None, 512])
    out_layer = input_layer
    out_layer = conv4_layer(out_layer)
    out_layer = conv5_layer(out_layer)
    resnet101_conv5 = tf.keras.Model(input_layer, out_layer)
    # print(resnet101_conv5.summary())

    return resnet101_conv3, resnet101_conv5


def create_ssd_layers():
    """ Create extra layers
        6th to 10th blocks
    """

    extra_layers = [
        # 6th block output shape: B, 512, 10, 10
        Sequential(layers=[
            Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(1024, 1, dilation_rate=1, strides=1, padding='same', activation='relu'),
            BatchNormalization(),
        ], name='block6'),

        # 6th block output shape: B, 512, 10, 10
        Sequential(layers=[
            Conv2D(256, 1, activation='relu', padding='same'),
            Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        ], name='block7'),

        # 7th block output shape: B, 256, 5, 5
        Sequential(layers=[
            Conv2D(128, 1, padding='same', activation='relu'),
            Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        ], name='block8'),

        # 8th block output shape: B, 256, 3, 3
        Sequential(layers=[
            Conv2D(128, 1, padding='same', activation='relu'),
            Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        ], name='block9'),

        # 9th block output shape: B, 256, 1, 1
        Sequential(layers=[
            Conv2D(128, 1, padding='same', activation='relu'),
            Conv2D(256, 3, strides=2, padding='valid', activation='relu'),
        ], name='block10')
    ]

    return extra_layers


def create_deconv_layer(module_num, fm_size, deconv_resolution):
    # print('module_num : ', module_num)
    # print('fm_size : ', fm_size)
    # print('deconv_resolution : ', deconv_resolution)
    layerName = layerNameCreater(baseName='deconv' + str(module_num + 1))
    if module_num == 0:
        before_input_resolution = 256
    else:
        before_input_resolution = 512

    before_input = Input(shape=(fm_size[0], fm_size[0], before_input_resolution))
    feature_input = Input(shape=(fm_size[1], fm_size[1], deconv_resolution))
    # print('before_input.get_shape().as_list() : ', before_input.get_shape().as_list())

    from_before = Conv2DTranspose(512, 2, strides=2, padding='valid', name=layerName.call('deconv'))(before_input)

    pad = 'valid' if from_before.get_shape().as_list()[1] != feature_input.get_shape().as_list()[1] else 'same'
    ks = 2 if pad == 'valid' else 3
    # pad = 'valid'
    # ks = 2

    from_before = Conv2D(512, ks, padding=pad, name=layerName.call('conv'))(from_before)
    before_output = BatchNormalization(name=layerName.call('bn'))(from_before)
    
    from_feature = Conv2D(512, 1, padding='same', name=layerName.call('conv'))(feature_input)
    from_feature = BatchNormalization(name=layerName.call('bn'))(from_feature)
    from_feature = Activation('relu', name=layerName.call('relu'))(from_feature)
    from_feature = Conv2D(512, 3, padding='same', name=layerName.call('conv'))(from_feature)
    feature_output = BatchNormalization(name=layerName.call('bn'))(from_feature)
    # print('before_output : ', before_output.get_shape().as_list())
    # print('feature_output : ', feature_output.get_shape().as_list())
    
    x = Multiply()([before_output, feature_output])
    # print('x : ', x.get_shape().as_list())
    x = Activation('relu', name=layerName.call('relu'))(x)
    # print('x activation : ', x.get_shape().as_list())
    # print()
    deconv_module = tf.keras.Model(inputs=[before_input, feature_input], outputs=x)

    return deconv_module


def create_prediction_layer(num_module):
    baseName = 'pred_' + str(num_module+1)
    layerName = layerNameCreater(baseName=baseName)
    if num_module == 0:
        input_resolution = 256
    else:
        input_resolution = 512
    input_layer = Input(shape=[None, None, input_resolution])
    shortcut = input_layer
    x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(input_layer)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(x)
    x = BatchNormalization(name=layerName.call('bn'))(x)
    x = Activation('relu', name=layerName.call('relu'))(x)
    x = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(x)

    shortcut = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(shortcut)
    shortcut = BatchNormalization(name=layerName.call('bn'))(shortcut)

    x = Add()([x, shortcut])
    output_layer = Activation('relu', name=layerName.call('relu'))(x)

    pred_module = tf.keras.Model(input_layer, output_layer)

    return pred_module


# def prediction_layer(x):
#     layerName = layerNameCreater(baseName='pred')

#     shortcut = x
    
#     x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(x)
#     x = BatchNormalization(name=layerName.call('bn'))(x)
#     x = Activation('relu', name=layerName.call('relu'))(x)
#     x = Conv2D(256, 1, strides=1, padding='same', name=layerName.call('conv'))(x)
#     x = BatchNormalization(name=layerName.call('bn'))(x)
#     x = Activation('relu', name=layerName.call('relu'))(x)
#     x = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(x)

#     shortcut = Conv2D(1024, 1, strides=1, padding='same', name=layerName.call('conv'))(shortcut)
#     shortcut = BatchNormalization(name=layerName.call('bn'))(shortcut)

#     x = Add()([x, shortcut])
#     x = Activation('relu', name=layerName.call('relu'))(x)

#     return x


def create_cls_head_layers(num_classes):
    """ Create layers for classification
    """
    layerName = layerNameCreater(baseName='conf')

    conf_head_layers = [
        Conv2D(anchors[0] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 4th block
        Conv2D(anchors[1] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 7th block
        Conv2D(anchors[2] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 8th block
        Conv2D(anchors[3] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 9th block
        Conv2D(anchors[4] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 10th block
        Conv2D(anchors[5] * num_classes, kernel_size=3, padding='same', name=layerName.call('conv')),  # for 11th block
        Conv2D(anchors[6] * num_classes, kernel_size=1, name=layerName.call('conv'))  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    layerName = layerNameCreater(baseName='loc')


    loc_head_layers = [
        Conv2D(anchors[0] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[1] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[2] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[3] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[4] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[5] * 4, kernel_size=3, padding='same', name=layerName.call('conv')),
        Conv2D(anchors[6] * 4, kernel_size=1)
    ]

    return loc_head_layers
