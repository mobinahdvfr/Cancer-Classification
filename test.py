import tensorflow as tf

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

# Define a simple convolution layer
conv_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", activation="relu")

# Create a dummy input tensor
input_tensor = tf.random.normal([1, 10, 10, 1])  # (Batch, Height, Width, Channels)

# Apply convolution
output_tensor = conv_layer(input_tensor)

print("Conv layer output shape:", output_tensor.shape)
