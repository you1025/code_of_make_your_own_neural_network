import numpy as np
from scipy.special import expit, logit

import os, sys
sys.path.append(os.path.dirname(__file__))
from functions import label_to_onehot, calc_inverse_weights, to_scaled_value, convert_target_list

class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, seed=1025):
        # 活性化関数
        self.activation_function = expit
        self.prime_activation_function = (lambda x: expit(x) * (1.0 - expit(x)))

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

    def get_reversed_inputs(self):
        reversed_inputs = [
            self._get_reversed_inputs_from_label(target_label).T[0]
            for target_label
            in np.arange(10)
        ]
        return (
            np.array(np.arange(10)),
            np.array(reversed_inputs)
        )

    def get_reversed_inputs_for_train(self):
        (target_labels, reversed_inputs) = self.get_reversed_inputs()

        rng = np.random.default_rng()
        train_data = []
        for idx in np.arange(10):
            target_label = target_labels[idx]
            reversed_input = reversed_inputs[idx]
            for _ in np.arange(10):
                train_data.append((
                    convert_target_list(target_label),
                    reversed_input + np.abs(rng.normal(0, 0.25, 784))
                ))
        return train_data

    def _get_reversed_inputs_from_label(self, target_label):
        # 最終層
        targets = label_to_onehot(target_label)
        final_inputs = logit(targets)

        # 中間層
        hidden_inverse_weights = calc_inverse_weights(self.w_o_h)
        hidden_outputs_raw = np.dot(self.w_o_h.T, (final_inputs - self.b_o))
        hidden_outputs = to_scaled_value(hidden_outputs_raw)
        hidden_inputs = logit(hidden_outputs)

        # 入力層
        input_inverse_weights = calc_inverse_weights(self.w_h_i)
        input_outputs_raw = np.dot(self.w_h_i.T, (hidden_inputs - self.b_h))
        inputs = to_scaled_value(input_outputs_raw)

        return inputs
