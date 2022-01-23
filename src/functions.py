import numpy as np
import gzip

SCALED_MIN = 0.001
SCALED_MAX = 0.999

def get_mnist_raw_data(path):
    with gzip.open(path, "rt") as f:
        train_data_list = f.readlines()

    def convert_target_and_data(data):
        splitted = np.asfarray(data.strip().split(","))
        return (splitted[0], splitted[1:])
    raw_trains = [
        convert_target_and_data(train_data)
        for train_data
        in train_data_list
    ]
    
    return raw_trains

def convert_target_list(idx: int):
    target_list = np.zeros(10) + SCALED_MIN
    target_list[idx] = SCALED_MAX
    return target_list

def convert_mnist_data(raw_data):
    return [
        (
            convert_target_list(int(raw[0])),
            raw[1] / 255 * SCALED_MAX + SCALED_MIN
        )
        for raw
        in raw_data
    ]

def nn_train(model, trains):
    for (target_list, train_list) in trains:
        model.train(train_list, target_list)

def calc_accuracy(model, tests):
    return np.array([
        np.argmax(target_list) == np.argmax(model.query(test_list))
        for (target_list, test_list)
        in tests
    ]).mean()

def to_scaled_value(x):
    x_min = np.min(x)
    x_max = np.max(x)
    return (SCALED_MAX - SCALED_MIN) * (x - x_min) / (x_max - x_min) + SCALED_MIN

def calc_inverse_weights(W):
    return np.array([ 1 / np.abs(w).sum() for w in W.T ], ndmin=2).T

def label_to_onehot(target):
    targets = np.zeros(10) + SCALED_MIN
    targets[target] = SCALED_MAX
    return np.array(targets, ndmin=2).T

def show_target_images(plt, images):
    fig = plt.figure(figsize=(15, 5))
    plt.subplots_adjust(hspace=0.25)

    for target_label in np.arange(10):
        ax = fig.add_subplot(2, 5, target_label+1)
        ax.imshow(images[target_label].reshape((28, 28)), interpolation="None")

    fig.tight_layout()