# Importing needed libraries

# OS - Library used for OS operations
import os

# SHUTIL - Library used for moving files
import shutil

# CV2 - Library used for computer vision operations
import cv2

# MATPLOTLIB - Library used for plotting graphs
import matplotlib.pyplot as plt

# NUMPY - Library used for tensor operations
import numpy as np

# TENSORFLOW - Library used for Deep Learning
import tensorflow as tf


def create_and_compile_model(input_shape=(224, 224, 3)):
    """Function that creates and compiles the deep learning model architecture

    Returns:
        [tensorflow.python.keras.engine.functional.Functional]: Model architecture
    """

    # Pretrained convolutional base from the MobileNetV2 architecture
    conv_base = tf.keras.applications.MobileNetV2(
        # Size of the image on which the model runs
        # Default input shape of the convolutional base is (224, 224) RGB image
        input_shape=input_shape,
    
        # Parameter that controls the width of the network (Width Multiplier in the official MobileNetV2 paper)
        # Default value 1 uses the number of filters at each layer from the paper
        alpha=1.0,
    
        # Parameter whether to include the fully connected layer
        # In our case we don't need the fully connected layer because we created our own
        include_top=False,
    
        # The actual weights of the convolutional base pre-trained on the ImageNet dataset
        weights='imagenet',
    
        # Optional keras input API, not used in our case
        input_tensor=None,
    
        # Applies Max Pooling
        pooling=max)
    
    # Freezing the weights of the convolutional base so as not to update them during backpropagation
    for layer in conv_base.layers:
        layer.trainable = False
    
    # Fully connected layers use the Keras Functional API
    
    # Flatten layer applied to the output of the convolutional base to flatten tensor
    flatten_layer = tf.keras.layers.Flatten()(conv_base.output)
    
    # First fully connected (dense) layer with 512 output neurons and activation function 'ReLu' applied to flatten layer
    dense_layer1 = tf.keras.layers.Dense(512, activation = 'relu')(flatten_layer)
    
    # First dropout layer to combat overfitting
    dropout_layer1 = tf.keras.layers.Dropout(0.5)(dense_layer1)
    
    # Second fully connected (dense) layer with 128 output neurons and 'ReLu' activation function applied on the dropout layer
    dense_layer2 = tf.keras.layers.Dense(128, activation = 'relu')(dropout_layer1)
    
    # Second dropout layer to combat overfitting
    dropout_layer2 = tf.keras.layers.Dropout(0.5)(dense_layer2)
    
    # Final output layer
    output_layer = tf.keras.layers.Dense(2, activation='softmax')(dropout_layer2)
    
    # Creating the actual model from the convolutional base and fully connected layers
    model = tf.keras.Model(inputs=conv_base.inputs, outputs=output_layer)
    
    # Compiling the model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def create_data_generator(rescale= 1./255):
    """Function that creates a data generator. 
       In our case we only need normalization, the images are already
       augmented.
       Other transformation can be applied.

    Args:
        rescale (float, optional): Used to normalize the images. Defaults to 1./255.

    Returns:
        [tensorflow.python.keras.preprocessing.image.ImageDataGenerator
        ]: Applies transformation to the images
    """
    data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=rescale)

    return data_generator


def prepare_dataset(dataset_path, subset, validation_split, image_size=(224, 224)):
    """Function to prepare the dataset for the model

    Args:
        dataset_path (String): Absolute path of the dataset
        subset (String): Whether the dataset subset is for training or for validation
        validation_split (float): Percentage of dataset to be used for validation
        image_size (tuple, optional): [description]. Defaults to (224, 224).

    Returns:
        tensorflow.python.data.ops.dataset_ops.BatchDataset: Dataset to fit the model to
    """
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        # Absolute path of the dataset
        dataset_path,

        # Labels are generated from the directory structure
        labels='inferred',

        # Labels are encoded as a categorical vector
        label_mode='categorical',

        # Class names
        class_names=['Green Light Augmented', 'Red Light Augmented'],

        # Color mode for the images
        color_mode='rgb',

        # Size of the batches of data
        batch_size=32,

        # Size to resize the images to after they are read from disk
        image_size=image_size,

        # Whether to shuffle the data
        shuffle=True,

        seed=0,

        # Fraction of data to reserve for validation
        validation_split=validation_split,

        # Wether the subset is for training or validation
        subset=subset
    )

    return dataset


def save_model(model, save_path):
    """Wrapper function to save the model.

    Args:
        model (tensorflow.python.keras.engine.functional.Functional): Trained model to be saved
        save_path (String): Absolute path where to save the model
    """
    tf.saved_model.save(model, save_path)


def convert_model_to_lite(original_model_path, lite_model_path):
    """Function to convert a trained model from storage to tensorflow lite version

    Args:
        original_model_path (String): Absolute path where the ordinary tensorflow model is saved
        lite_model_path (String): Absolute path where the tensorflow lite model is saved. 
    """

    # Create tensorflow lite convertor to convert the basic tensorflow model
    lite_converter = tf.lite.TFLiteConverter.from_saved_model(original_model_path)

    # Metric that the converter should optimize the model for
    lite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

    lite_model = lite_converter.convert()

    # Saving the tensorflow lite model
    with open(os.path.join(lite_model_path, 'lite_model.tflite'), 'wb') as f:
        f.write(lite_model)
    

def main():

    print('Creating model...')

    # Creating and compiling the model architecture
    model = create_and_compile_model()

    print('Preparing dataset...')

    # Dataset path
    dataset_path = r'D:\Proiect SI\Dataset'


    # Creating the directory for the dataset
    if not os.path.isdir(r'D:\Proiect SI\Dataset'):
        os.mkdir(r'D:\Proiect SI\Dataset')
    
    # Moving the augmented images directories into the dataset directory
    dest = shutil.move(r'D:\Proiect SI\Green Light Augmented', r'D:\Proiect SI\Dataset')
    dest2 = shutil.move(r'D:\Proiect SI\Red Light Augmented', r'D:\Proiect SI\Dataset' )

   

    # Creating the training data generator that normalize the input images in the range [0, 1]
    train_datagen = create_data_generator()

    # Creating the validation data generator that normalize the input images in the range [0, 1]
    validation_datagen = create_data_generator()


    # Preparing dataset for the model
    train_dataset = prepare_dataset(dataset_path, 'training', 0.2)
    validation_dataset = prepare_dataset(dataset_path, 'validation', 0.2)

    print('Training model...')

    # Training model
    history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset, verbose=1)

    # Path where to save the tensorflow model to
    save_path = r'D:\Proiect SI\basic_model'

    # Saving the model
    save_model(model, save_path)

    save_path = r'D:\Proiect SI\basic_model'

    # Path where to save the tensorflow lite model to
    lite_model_savepath = r'D:\Proiect SI\tf_lite'

    if not os.path.exists(lite_model_savepath):
        os.mkdir(lite_model_savepath)

    print('Converting and saving tensorflow lite model...')


    convert_model_to_lite(save_path, lite_model_savepath)


if __name__ == '__main__':
    main()