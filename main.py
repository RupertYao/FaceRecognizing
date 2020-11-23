from LoggingFace import loggingFromVideo
from Predict import recognizeFromVideo
from Train import trainModel


def logging(tag):
    loggingFromVideo(tag)


def train():
    trainModel()


def predict():
    recognizeFromVideo()


if __name__ == "__main__":
    train()
    recognizeFromVideo()
