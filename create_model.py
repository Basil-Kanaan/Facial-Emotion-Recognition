import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import Rescaling


num_classes = 7


def setup_model():

    # initialize Sequential model with layers for image recog.
    # input is 48x48 pixel image with 7-element array output
    _model = tf.keras.Sequential([
        Rescaling(1. / 255, input_shape=(48, 48, 1)),
        Conv2D(16, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes)
    ])

    _model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

    return _model


def preprocess_data(path):

    # turn image directory into tensorflow dataset, prepared for use
    _data = tf.keras.preprocessing.image_dataset_from_directory(path, color_mode='grayscale', image_size=(48, 48))
    _data = _data.cache().shuffle(1000).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return _data


if __name__ == "__main__":

    # get train and validation data
    dataTrain = preprocess_data("data\\train")
    dataVal = preprocess_data("data\\val")

    # initialize and fit model to training data
    model = setup_model()
    model.fit(dataTrain, validation_data=dataVal, epochs=15)

    # save the model to a new folder if it doesn't already exist, otherwise overwrite it
    model.save("model_save")
