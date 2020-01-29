#from .resnet import (
#    _handle_dim_ordering,
#    _get_block,_bn_relu,
#    _conv_bn_relu,MaxPooling2D,
#    _residual_block,
#    AveragePooling2D,
#    basic_block,
#    bottleneck,
#    _bn_InvRelu,
#    _conv_bn
#)

from . import resnet
#from .dual_relu_resnet import(
#    dual_relu_basic_block,
#    dual_relu_bottleneck,
#    dual_relu_residual
#)

from . import dual_relu_resnet

#from .Invert_ReLu_resnet import _invert_bn_relu
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
from keras.layers import GlobalAveragePooling2D

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions,option):
        """Builds a custom ResNet like architecture.

        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved

        Returns:
            The keras `Model`.
        """
        AXIS = resnet._handle_dim_ordering()
        ROW_AXIS = AXIS[0]
        COL_AXIS = AXIS[1]

        #options
        inv_relu = bool(option["relu_option"])
        double_input = bool(option["double_input"])
        default_filters=int(option["filters"])
        wide = bool(option["wide"])

        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = resnet._get_block(block_fn)
        input = Input(shape=input_shape)

        #conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        if wide == False:
            conv1 = resnet._conv_bn(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        else:
            conv1 = resnet._conv_bn(filters=16,kernel_size=(3,3),strides=(1,1))(input)

        if (block_fn == resnet.basic_block) or (block_fn == resnet.bottleneck) :
            if inv_relu == True:
                invert_positive = Lambda(lambda x: x*(-1))(conv1)
                relu_positive = Activation("relu")(invert_positive)
            else:
                relu_positive = Activation("relu")(conv1)
        else:
            relu_positive = Activation("relu")(conv1)

        if double_input == True:
            invert = Lambda(lambda x: x*(-1))(conv1)
            relu_negative = Activation("relu")(invert)

        def f(x):
            if wide == False:   
                pool1 = resnet.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
                block = pool1
            else:
                block=x
            filters = default_filters
            #filters=64
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
            #pool2 = resnet.AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                    #strides=(1, 1))(block)
            return block
        
        if double_input == False:
            main_model = f(relu_positive)
            #flatten1 = Flatten()(main_model)
            global_pool = GlobalAveragePooling2D()(main_model)
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                        activation="softmax")(global_pool)
            model = Model(inputs=input, outputs=dense)
            
            return model
        else:
            positive_model = f(relu_positive)
            negative_model = f(relu_negative)
            concat = Concatenate(axis=3)([positive_model,negative_model])
            #flatten1 = Flatten()(concat)
            global_pool = GlobalAveragePooling2D()(concat)
            dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                        activation="softmax")(global_pool)
            model = Model(inputs=input, outputs=dense)

            return model
    
    @staticmethod
    def build_manual(input_shape, num_outputs, option):
        block = option["block"]

        if block == "basic_block":
            block_fn = resnet.basic_block
        elif block == "bottleneck":
            block_fn = resnet.bottleneck
        elif block == "double_basic":
            block_fn = dual_relu_resnet.dual_relu_basic_block
        elif block == "double_bottleneck":
            block_fn = dual_relu_resnet.dual_relu_bottleneck
            
            
        #return ResnetBuilder.build(input_shape, num_outputs,block_fn, [2,2,2,2], option = option)
        return ResnetBuilder.build(input_shape, num_outputs,block_fn, option["reseption"], option = option)

'''
    @staticmethod
    def build_resnet_18(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2],option = option)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3],option = option)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3],option = option)

    @staticmethod
    def build_resnet_101(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3],option = option)

    @staticmethod
    def build_resnet_152(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3],option = option)
    
    @staticmethod
    def build_dualresnet_18(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False, "concatenate":False}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [2, 2, 2, 2],option = option)

    @staticmethod
    def build_dualresnet_34(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":False}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [3, 4, 6, 3],option = option)

    @staticmethod
    def build_dualresnet_50(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":False}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 6, 3],option = option)

    @staticmethod
    def build_dualresnet_101(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":False}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 23, 3],option = option)

    @staticmethod
    def build_dualresnet_152(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":False}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 8, 36, 3],option = option)

    #------------------------
    # option is true models
    #------------------------

    @staticmethod
    def build_invert_relu_resnet_18(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": True, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], option = option)

    @staticmethod
    def build_invert_relu_resnet_34(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": True, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3], option = option)

    @staticmethod
    def build_invert_relu_resnet_50(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": True, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3], option = option)

    @staticmethod
    def build_invert_relu_resnet_101(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": True, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3], option = option)

    @staticmethod
    def build_invert_relu_resnet_152(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": True, "double_input": False}
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3], option = option)
    
    @staticmethod
    def build_concatenate_dualresnet_18(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":True}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [2, 2, 2, 2], option = option)

    @staticmethod
    def build_concatenate_dualresnet_34(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":True}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_basic_block, [3, 4, 6, 3], option = option)

    @staticmethod
    def build_concatenate_dualresnet_50(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":True}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 6, 3], option = option)

    @staticmethod
    def build_concatenate_dualresnet_101(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":True}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 4, 23, 3], option = option)

    @staticmethod
    def build_concatenate_relu_dualresnet_152(input_shape, num_outputs, option = None):
        if option is None:
            option = {"relu_option": False, "double_input": False,"concatenate":True}
        return ResnetBuilder.build(input_shape, num_outputs, dual_relu_bottleneck, [3, 8, 36, 3], option = option)
'''