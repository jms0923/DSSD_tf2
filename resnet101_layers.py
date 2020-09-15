import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Input, BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Multipy, Add

anchors = [4, 6, 6, 6, 4, 4, 4]


def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 

def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x


def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x


def conv4_layer(x):
    shortcut = x        
  
    for i in range(23):
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x


def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=(2, 2), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x


def create_resnet101_layers():
    input_layer = Input(shape=[None, None, 3])
    out_layer = input_layer
    out_layer = conv1_layer(out_layer)
    out_layer = conv2_layer(out_layer)
    out_layer = conv3_layer(out_layer)
    resnet101_conv3 = tf.keras.Model(input_layer, out_layer)


    input_layer = Input(shape=[None, None, 3])
    out_layer = input_layer
    out_layer = conv4_layer(out_layer)
    out_layer = conv5_layer(out_layer)
    resnet101_conv5 = tf.keras.Model(input_layer, out_layer)

    return resnet101_conv3, resnet101_conv5


def ssd_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    extra_layers = [
        # 6th block output shape: B, 512, 10, 10
        block6 = Sequential(layers=[
            layers.Conv2D(1024, 3, dilation_rate=6, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(1024, 1, dilation_rate=1, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ], name='block3')

        # 6th block output shape: B, 512, 10, 10
        block7 = Sequential(layers=[
            layers.Conv2D(256, 1, activation='relu', padding='same'),
            layers.Conv2D(512, 3, strides=2, padding='same', activation='relu'),
        ], name='block7')

        # 7th block output shape: B, 256, 5, 5
        block8 = Sequential(layers=[
            layers.Conv2D(128, 1, padding='same', activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        ], name='block8')

        # 8th block output shape: B, 256, 3, 3
        block9 = Sequential(layers=[
            layers.Conv2D(128, 1, padding='same', activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same', activation='relu'),
        ], name='block9')

        # 9th block output shape: B, 256, 1, 1
        block10 = Sequential(layers=[
            layers.Conv2D(128, 1, padding='same', activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='valid', activation='relu'),
        ], name='block10')
    ]

    return extra_layers


# have to add head layer, location layer
def prediction_layer(x):
    shortcut = x
    
    x = Conv2D(256, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, 1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1024, 1, strides=1, padding='same')(x)

    shortcut = Conv2D(1024, 1, strides=1, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        Conv2D(anchors[0] * num_classes, kernel_size=3, padding='same'),  # for 4th block
        Conv2D(anchors[1] * num_classes, kernel_size=3, padding='same'),  # for 7th block
        Conv2D(anchors[2] * num_classes, kernel_size=3, padding='same'),  # for 8th block
        Conv2D(anchors[3] * num_classes, kernel_size=3, padding='same'),  # for 9th block
        Conv2D(anchors[4] * num_classes, kernel_size=3, padding='same'),  # for 10th block
        Conv2D(anchors[5] * num_classes, kernel_size=3, padding='same'),  # for 11th block
        Conv2D(anchors[6] * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        Conv2D(anchors[0] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[1] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[2] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[3] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[4] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[5] * 4, kernel_size=3, padding='same'),
        Conv2D(anchors[6] * 4, kernel_size=1)
    ]

    return loc_head_layers


def deconv_layer(from_before, from_feature):
    from_before = Conv2DTranspose(512, 2, strides=2, padding='valid')(from_before)
    pad = 'valid' if from_before.get_shape().as_list()[1] != from_feature.get_shape().as_list()[1] else 'same'
    ks = 2 if pad == 'valid' else 3
    from_before = Conv2D(512, ks, padding=pad)(from_before)
    from_before = BatchNormalization()(from_before)
    
    from_feature = Conv2D(512, 1, padding='same')(from_feature)
    from_feature = BatchNormalization()(from_feature)
    from_feature = Activation('relu')(from_feature)
    from_feature = Conv2D(512, 3, padding='same')(from_feature)
    from_feature = BatchNormalization()(from_feature)
    
    x = Multiply()([from_before, from_feature])
    x = Activation('relu')(x)

    return x
