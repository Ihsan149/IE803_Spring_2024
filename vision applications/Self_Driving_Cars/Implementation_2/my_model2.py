import tensorflow as tf
from tensorflow.keras.applications import ResNet18
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout

# Load the pre-trained ResNet-18 model without the top classification layer
base_model = ResNet18(weights='imagenet', include_top=False, input_shape=(66, 200, 3))

# Freeze the pre-trained layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers on top of the pre-trained model
x = Flatten()(base_model.output)
x = Dense(1164, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(50, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='linear')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print model summary
model.summary()
