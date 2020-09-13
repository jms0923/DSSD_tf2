import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Input, BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Multipy, Add


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
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), dilation_rate=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
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


# have to add head layer, location layer
def prediction_layer(x):
    shortcut = x
    
    x = Conv2D(256, 1, strides=1, padding='same')(x)
    x = Conv2D(256, 1, strides=1, padding='same')(x)
    x = Conv2D(1024, 1, strides=1, padding='same')(x)

    shortcut = Conv2D(1024, 1, strides=1, padding='same')(shortcut)

    x = Add()([x, shortcut])

    return x


def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 11th block
        layers.Conv2D(4 * num_classes, kernel_size=1)  # for 12th block
    ]

    return conf_head_layers


def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=1)
    ]

    return loc_head_layers


# have to modify padding and strides
def from_feature_layer(x):
    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    return x


# have to modify padding and strides
def from_before_layer(x):
    x = Conv2DTranspose(512, 2, strides=2, padding='valid')(x)
    x = Conv2D(512, 3, strides=1, padding='valid')(x)
    x = BatchNormalization()(x)


def deconv_layer(from_deconv, from_feature):
    fdl = from_deconv_layer(from_decon)
    ffl = from_before_layer(from_feature)
    x = Multipy()[fdl, ffl]
    x = Activation('relu')(x)

    return x
