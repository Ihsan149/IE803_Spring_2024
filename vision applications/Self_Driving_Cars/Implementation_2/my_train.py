import os
import tensorflow as tf
from my_driving_data import LoadTrainBatch, LoadValBatch, num_images
from my_model import model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from datetime import datetime

LOGDIR = './logs'

L2NormConst = 0.001

# Compile the model
optimizer = Adam(learning_rate=1e-4)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Tensorboard callback
logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Checkpoint callback
checkpoint_path = os.path.join(LOGDIR, "model.{epoch:02d}-{val_loss:.2f}.h5")
checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1)

# Train the model
epochs = 30
batch_size = 100

for epoch in range(epochs):
    for i in range(int(num_images / batch_size)):
        xs, ys = LoadTrainBatch(batch_size)
        model.train_on_batch(xs, ys)

        if i % 10 == 0:
            xs, ys = LoadValBatch(batch_size)
            loss_value = model.evaluate(xs, ys, verbose=0)
            print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    if epoch % 1 == 0:
        model.save_weights(checkpoint_path.format(epoch=epoch, val_loss=loss_value))
        print("Model saved in file: %s" % checkpoint_path.format(epoch=epoch, val_loss=loss_value))

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
