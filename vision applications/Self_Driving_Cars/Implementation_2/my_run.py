import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
import cv2
import numpy as np
import os


# Define the model architecture
def weight_variable(shape, dtype=None, partition_info=None):
    return tf.random.truncated_normal(shape, stddev=0.1, dtype=dtype)


def bias_variable(shape, dtype=None, partition_info=None):
    return tf.constant(0.1, shape=shape, dtype=dtype)


x_input = Input(shape=(66, 200, 3))
conv1 = Conv2D(24, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable,
               bias_initializer=bias_variable)(x_input)
conv2 = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable,
               bias_initializer=bias_variable)(conv1)
conv3 = Conv2D(48, (5, 5), strides=(2, 2), padding='valid', activation='relu', kernel_initializer=weight_variable,
               bias_initializer=bias_variable)(conv2)
conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=weight_variable,
               bias_initializer=bias_variable)(conv3)
conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding='valid', activation='relu', kernel_initializer=weight_variable,
               bias_initializer=bias_variable)(conv4)

flatten = Flatten()(conv5)

fc1 = Dense(1164, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(flatten)
fc1_drop = Dropout(0.5)(fc1)
fc2 = Dense(100, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc1_drop)
fc2_drop = Dropout(0.5)(fc2)
fc3 = Dense(50, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc2_drop)
fc3_drop = Dropout(0.5)(fc3)
fc4 = Dense(10, activation='relu', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc3_drop)
fc4_drop = Dropout(0.5)(fc4)

output = Dense(1, activation='linear', kernel_initializer=weight_variable, bias_initializer=bias_variable)(fc4_drop)

model = Model(inputs=x_input, outputs=output)
print(model.summary())

# Load the saved weights into the model
model.load_weights("logs/model.h5")

# Path to the directory containing the images
image_dir = "/media/ihsan/HDDP12/Workspace/End-to-End-Learning-for-Self-Driving-Cars/Implementation_2/07012018/data"
# Path to the directory containing the images
image_dir = "/media/ihsan/HDDP12/Workspace/End-to-End-Learning-for-Self-Driving-Cars/Implementation_2/driving_dataset"

# Load the steering wheel image
img = cv2.imread('New Project.jpg', 0)

# Get dimensions of the steering wheel image
rows, cols = img.shape

smoothed_angle = 0

# Iterate over images in the directory
for i in range(42406, 45568): # Assuming images are numbered from 0.jpg to 45567.jpg
    # Construct the file path for the current image
    image_path = os.path.join(image_dir, f"{i}.jpg")

    # Check if the image file exists
    if os.path.exists(image_path):
        # Read the image
        frame = cv2.imread(image_path)

        # Resize the frame
        resized_frame = cv2.resize(frame, (200, 66))

        # Convert frame to float32 and normalize
        image = np.array(resized_frame, dtype=np.float32) / 255.0

        # Make prediction using the Keras model
        degrees = model.predict(np.expand_dims(image, axis=0))[0][0] * 180 / np.pi

        print("Predicted steering angle: " + str(degrees) + " degrees")



        # Make smooth angle transitions for the steering wheel image
        smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(
            degrees - smoothed_angle)
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("steering wheel", dst)# Show the current frame
        cv2.imshow('frame', frame)

        # Check for user input to quit the loop
        if cv2.waitKey(70) == ord('q'):
            break
    else:
        print(f"Image {i}: File not found.")

cv2.destroyAllWindows()
