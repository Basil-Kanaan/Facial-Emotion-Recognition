import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys

from tensorflow.keras.layers.experimental.preprocessing import Resizing


emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


if __name__ == "__main__":

    # load model from it's save folder
    model = tf.keras.models.load_model('model_save')

    # load image from path as second arg of command line
    # (img must be 1:1 ratio for this program to work properly)
    img = tf.keras.preprocessing.image.load_img(sys.argv[1], color_mode="grayscale", target_size=(48, 48))

    # 2nd arg may be provided in command line to select interpolation
    interpol = "mitchellcubic"  # default
    if len(sys.argv) == 3:
        interpol = sys.argv[2].strip()

    # convert img to array and resize it to be 48x48 pixels.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = Resizing(48, 48, interpolation=interpol)(tf.expand_dims(img_array, 0))

    # show the resized, grayscale image
    plt.imshow(np.reshape(img_array, (48, 48)), cmap="gray")
    plt.show()

    # make a prediction and use softmax to get probabilities of classes
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # indices of classes with top 3 highest probabilities
    top_3 = np.argsort(score)[-3:][::-1]

    print(f"You are most likely {str(emotions[np.argmax(score)]).lower()} with a {100 * np.max(score):.2f}% confidence.")
    print(f"The 2nd & 3rd likelihoods are {emotions[top_3[1]]} ({100 * score[top_3[1]]:.2f}%) "
          f"&  {emotions[top_3[2]]} ({100 * score[top_3[2]]:.2f}%)")
