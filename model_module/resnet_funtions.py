from . import resnet
from . import dual_relu_resnet
from . import Invert_ReLu_resnet
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Concatenate,
    Lambda
)
from keras import backend as K
from keras.models import Model

def normal_resnet(block_fn,option,repetitions):
    inv_relu = bool(option["relu_option"])
    AXIS = resnet._handle_dim_ordering()
    ROW_AXIS = AXIS[0]
    COL_AXIS = AXIS[1]

    def f(x):    
        pool1 = resnet.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = resnet._residual_block(block_fn, filters=filters, repetitions=r, option=option,is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        if (block_fn == resnet.basic_block) or (block_fn == resnet.bottleneck):
            if inv_relu:
                block = resnet._invert_bn_relu(block)
            else:
                block = resnet._bn_relu(block)
        else:
            block = resnet._bn_relu(block)

        # Classifier block
        #ROW_AXIS = _handle_dim_ordering().ROW_AXIS
        #COL_AXIS = _handle_dim_ordering().COL_AXIS

        block_shape = K.int_shape(block)
        pool2 = resnet.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                strides=(1, 1))(block)
        return pool2
        
    return f