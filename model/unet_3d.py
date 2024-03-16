# model/unet_3d.py
import tensorflow as tf
from tensorflow.keras import layers

def conv_block(input_tensor, num_filters):
    encoder = layers.Conv3D(num_filters, (3, 3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv3D(num_filters, (3, 3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([decoder, concat_tensor], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = conv_block(decoder, num_filters)
    return decoder

def unet_3d(inputs, num_filters=64):
    # Encoder
    encoder_pool1, encoder1 = encoder_block(inputs, num_filters)
    encoder_pool2, encoder2 = encoder_block(encoder_pool1, num_filters*2)
    encoder_pool3, encoder3 = encoder_block(encoder_pool2, num_filters*4)
    encoder_pool4, encoder4 = encoder_block(encoder_pool3, num_filters*8)
    
    # Center
    center = conv_block(encoder_pool4, num_filters*16)
    
    # Decoder
    decoder4 = decoder_block(center, encoder4, num_filters*8)
    decoder3 = decoder_block(decoder4, encoder3, num_filters*4)
    decoder2 = decoder_block(decoder3, encoder2, num_filters*2)
    decoder1 = decoder_block(decoder2, encoder1, num_filters)
    
    # Output
    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid')(decoder1)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# Usage
# Define input layer
input_shape = (128, 128, 128, 1) # Example input shape, change as needed
inputs = tf.keras.Input(input_shape)

# Create model
model = unet_3d(inputs)
model.summary()

