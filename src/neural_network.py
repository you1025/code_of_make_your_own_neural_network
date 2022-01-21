from typing import final
import numpy as np
from scipy.special import expit

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, seed=1025):
        # 活性化関数
        self.activation_function = expit
        self.prime_activation_function = (lambda x: expit(x) * (1.0 - expit(x)))
#        self.activation_function       = np.vectorize(lambda x: 0.0 if x <= 0.0 else x)
#        self.prime_activation_function = np.vectorize(lambda x: 0.0 if x <= 0.0 else 1.0)

        # 各レイヤのノード数
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 学習率
        self.learning_rate = learning_rate

        # リンクの初期化
        rng = np.random.default_rng(seed)
        self.w_h_i = rng.normal(0.0, hidden_nodes**(-1/2), size=(hidden_nodes, input_nodes))
        self.w_o_h = rng.normal(0.0, output_nodes**(-1/2), size=(output_nodes, hidden_nodes))

        # バイアスの初期化
        self.b_h = rng.uniform(-0.5, 0.5, hidden_nodes).reshape((hidden_nodes, 1))
        self.b_o = rng.uniform(-0.5, 0.5, output_nodes).reshape((output_nodes, 1))

    def train(self, input_list, target_list):
        targets = np.array(target_list, ndmin=2).T

        # 順伝播
        (inputs, hidden_inputs, hidden_outputs, final_inputs, final_outputs) = self._feed_forward(input_list)

        # 逆伝播(最終層の更新量を算出)
        output_unit_errors = -2 * (targets - final_outputs) * self.prime_activation_function(final_inputs)
        delta_w_o_h = self.learning_rate * np.dot(output_unit_errors, hidden_outputs.T)
        delta_b_o   = self.learning_rate * output_unit_errors

        # 逆伝播(中間層の更新量を算出)
        hidden_unit_errors = np.dot(self.w_o_h.T, output_unit_errors) * self.prime_activation_function(hidden_inputs)
        delta_w_h_i = self.learning_rate * np.dot(hidden_unit_errors, inputs.T)
        delta_b_h   = self.learning_rate * hidden_unit_errors

        # 学習の実行
        self.w_o_h -= delta_w_o_h
        self.b_o   -= delta_b_o
        self.w_h_i -= delta_w_h_i
        self.b_h   -= delta_b_h

    def query(self, input_list):
        return self._feed_forward(input_list)[4]

    def _feed_forward(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.w_h_i, inputs) + self.b_h
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.w_o_h, hidden_outputs) + self.b_o
        final_outputs = self.activation_function(final_inputs)

        return (inputs, hidden_inputs, hidden_outputs, final_inputs, final_outputs)


def get_mnist_raw_data(path):
    with open(path, "r") as f:
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

def convert_mnist_data(raw_data):
    def convert_target_list(idx: int):
        target_list = np.zeros(10) + 0.01
        target_list[idx] = 0.99
        return target_list

    return [
        (
            convert_target_list(int(raw[0])),
            raw[1] / 255 * 0.99 + 0.01
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