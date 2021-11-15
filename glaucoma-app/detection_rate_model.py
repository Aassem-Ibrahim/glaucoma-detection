from keras.models import load_model
import numpy as np


def normalize_sample_data(sample):
    # extract cup and disc from sample
    cup, disc = sample['cup'], sample['disc']
    # get normalize vector
    norm = max(disc['h'], disc['w'])
    # normalize cup info
    cup['x'] = (cup['x'] + cup['w'] / 2 - disc['x']) / norm
    cup['y'] = (cup['y'] + cup['h'] / 2 - disc['y']) / norm
    cup['w'] /= norm
    cup['h'] /= norm
    cup['a'] /= norm * norm
    # normalize disc info
    disc['x'] = disc['w'] / 2 / norm
    disc['y'] = disc['h'] / 2 / norm
    disc['w'] /= norm
    disc['h'] /= norm
    disc['a'] /= norm * norm
    # return normalized sample data
    return sample


class DetectionRateModel:
    """ Detection Rate Model (CDR)
        This model is used with all cup/disc 10 parameters.
    """
    # model container
    model = None

    def __init__(self, model_filename):
        try:
            # See if file exists
            open(model_filename, 'r')
            # Load CDR model
            self.model = load_model(model_filename)
        except FileNotFoundError:
            # Display Error
            print(f"ERROR: Model '{model_filename}' is not found")

    def predict(self, case):
        # Check if model is loaded correctly
        if self.model is not None:
            return self.model.predict(np.array([case]))
        else:
            # return model error
            return [(-1, -1)]


if __name__ == '__main__':
    CDR_MODEL = 'models/detection_rate_model.h5'
    cdr_model = DetectionRateModel(CDR_MODEL)
