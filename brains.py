#TODO Core logic of the neural network/  Convolutional neural networks (CNNs) - good with images
import tensorflow as tf
from keras.utils import plot_model

height = 1024  # Replace with your image height
width = 768   # Replace with your image width
channels = 3  # For RGB images

import tensorflow as tf

# Define model dimensions
# For resources (18 types + 1 for no resource)
resource_classes = 18 + 1

import tensorflow as tf

# Define model dimensions
image_height = 720  # Adjust to the height of your full screenshot
image_width = 1280  # Adjust to the width of your full screenshot
channels = 3  # Color images (RGB)

# Define the number of classes for terrain and resources
terrain_classes = 3  # For example: plains, snow, desert
resource_classes = 19  # 18 resource types + 1 for no resource

# Define the number of grid locations in the image (assuming a fixed grid structure)#TODO Change the approach
num_grid_locations = 30  # The number of distinguishable grid locations in the screenshot

# Define the CNN model#TODO consider Faster R-CNN model or example in the test.py
model = tf.keras.Sequential([
    # Convolutional layers for feature extraction
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, channels)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten the convolutional features into a single vector
    tf.keras.layers.Flatten(),

    # Dense layers for feature interpretation
    tf.keras.layers.Dense(1024, activation='relu'),

    # Output layers for classification
    # Assuming a grid structure where each grid cell's information is concatenated
    tf.keras.layers.Dense(num_grid_locations * terrain_classes, activation='softmax', name='terrain_output'),
    tf.keras.layers.Dense(num_grid_locations * resource_classes, activation='softmax', name='resource_output')
])

# Compile the model
model.compile(optimizer='adam',
              loss={
                  'terrain_output': 'categorical_crossentropy',
                  'resource_output': 'categorical_crossentropy'
              },
              metrics=['accuracy'])

# Model summary
model.summary()
plot_model(model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)