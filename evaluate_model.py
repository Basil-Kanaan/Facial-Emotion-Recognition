import tensorflow as tf
from create_model import preprocess_data


emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


if __name__ == "__main__":

    # load model from it's save folder
    model = tf.keras.models.load_model('model_save')

    # load test data
    dataTest = preprocess_data("data\\test")

    # average accuracy
    sum = 0
    for i in range(500):
        Xtest, Ttest = next(dataTest.as_numpy_iterator())
        predictions = tf.math.argmax(model.predict(Xtest), axis=1)
        correct_predictions = tf.math.equal(predictions, Ttest)
        sum += tf.math.reduce_mean(tf.cast(correct_predictions, tf.float32))

    print(f"Average accuracy: {sum/500}")
