import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.activations import relu
from tensorflow.keras.constraints import max_norm

def weight_variable(shape, dtype=None, partition_info=None):
    return tf.random.truncated_normal(shape, stddev=0.1, dtype=dtype)

def bias_variable(shape, dtype=None, partition_info=None):
    return tf.constant(0.1, shape=shape, dtype=dtype)

x_input = Input(shape=(66, 200, 3))
# first convolutional layer
conv1 = Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(x_input)
# second convolutional layer
conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(conv1)
# third convolutional layer
conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(conv2)
# fourth convolutional layer
conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(conv3)
# fifth convolutional layer
conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(conv4)

flatten = Flatten()(conv5)

# fully connected layers
fc1 = Dense(1164, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(flatten)
fc1_drop = Dropout(0.5)(fc1)
fc2 = Dense(100, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc1_drop)
fc2_drop = Dropout(0.5)(fc2)
fc3 = Dense(50, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc2_drop)
fc3_drop = Dropout(0.5)(fc3)
fc4 = Dense(10, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc3_drop)
fc4_drop = Dropout(0.5)(fc4)

# output layer
output = Dense(1, activation='linear', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc4_drop)

model = Model(inputs=x_input, outputs=output)

model.summary()
