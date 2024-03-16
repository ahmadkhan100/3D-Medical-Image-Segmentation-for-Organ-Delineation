# model/optimizer.py
import tensorflow as tf

def optimize_model(model, save_path):
    # Convert Keras model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

# Usage
# Provide path to save the optimized model
optimized_model_path = 'path/to/save/optimized_model.tflite'

# Assuming 'model' is already created using the unet_3d.py script
optimize_model(model, optimized_model_path)

