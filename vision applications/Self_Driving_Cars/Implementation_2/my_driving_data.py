import random
import numpy as np
from PIL import Image
from skimage.transform import resize

xs = []
ys = []

# Points to the end of the last batch
train_batch_pointer = 0
val_batch_pointer = 0

# Read data.txt
with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        # Convert steering angle to radians
        ys.append(float(line.split()[1]) * np.pi / 180)

# Get number of images
num_images = len(xs)

# Shuffle list of images
c = list(zip(xs, ys))
random.shuffle(c)
xs, ys = zip(*c)

# Split data into training and validation sets
train_xs = xs[:int(len(xs) * 0.8)]
train_ys = ys[:int(len(xs) * 0.8)]

val_xs = xs[-int(len(xs) * 0.2):]
val_ys = ys[-int(len(xs) * 0.2):]

num_train_images = len(train_xs)
num_val_images = len(val_xs)

def LoadTrainBatch(batch_size):
    global train_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        image_path = train_xs[(train_batch_pointer + i) % num_train_images]
        image = Image.open(image_path)
        # Crop the bottom part of the image
        image = image.crop((0, image.size[1] - 150, image.size[0], image.size[1]))
        # Resize the image
        image = image.resize((200, 66))
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        x_out.append(image)
        y_out.append([train_ys[(train_batch_pointer + i) % num_train_images]])
    train_batch_pointer += batch_size
    # Convert to numpy array and add batch dimension
    return np.array(x_out), np.array(y_out)

def LoadValBatch(batch_size):
    global val_batch_pointer
    x_out = []
    y_out = []
    for i in range(batch_size):
        image_path = val_xs[(val_batch_pointer + i) % num_val_images]
        image = Image.open(image_path)
        # Crop the bottom part of the image
        image = image.crop((0, image.size[1] - 150, image.size[0], image.size[1]))
        # Resize the image
        image = image.resize((200, 66))
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        x_out.append(image)
        y_out.append([val_ys[(val_batch_pointer + i) % num_val_images]])
    val_batch_pointer += batch_size
    # Convert to numpy array and add batch dimension
    return np.array(x_out), np.array(y_out)

