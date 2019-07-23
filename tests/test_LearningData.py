from kde.learning_data import *
import numpy as np


def test_LearningData():
    features = np.arange(24).reshape(2, 3, 4)
    labels = np.arange(2)
    text_label = {0: '0', 1: '1'}
    data = LearningData('test', features, labels, text_labels)
    assert data.features.shape == (2, 12)
