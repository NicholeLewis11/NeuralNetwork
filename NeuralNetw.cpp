#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

MatrixXd sigmoid(const MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}

MatrixXd sigmoid_derivative(const MatrixXd& x) {
    return x.array() * (1.0 - x.array());
}

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layers) {
        for (int i = 1; i < layers.size(); i++) {
            weights_.emplace_back(MatrixXd::Random(layers[i], layers[i-1]));
            biases_.emplace_back(MatrixXd::Random(layers[i], 1));
        }
    }

    MatrixXd predict(const MatrixXd& inputs) {
        MatrixXd a = inputs;
        for (int i = 0; i < weights_.size(); i++) {
            a = sigmoid((weights_[i] * a) + biases_[i]);
        }
        return a;
    }

    void train(const MatrixXd& inputs, const MatrixXd& labels, double learning_rate) {
        std::vector<MatrixXd> activations;
        activations.emplace_back(inputs);

        // forward pass
        for (int i = 0; i < weights_.size(); i++) {
            MatrixXd z = (weights_[i] * activations[i]) + biases_[i];
            MatrixXd a = sigmoid(z);
            activations.emplace_back(a);
        }
