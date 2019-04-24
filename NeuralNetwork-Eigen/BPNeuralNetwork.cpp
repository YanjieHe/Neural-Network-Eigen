#include "BPNeuralNetwork.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
using namespace std;

BPNeuralNetwork::BPNeuralNetwork(int numOfInput, vector<int> numOfHidden,
                                 int numOfOutput)
    : input(numOfInput), hidden(numOfHidden.size()), output(numOfOutput),
      weights(numOfHidden.size() + 1), biases(numOfHidden.size() + 1),
      deltas(numOfHidden.size() + 1)
{
    if (numOfHidden.size() < 1)
    {
        throw string("There should be at least one hidden layer");
    }
    else
    {
        weights.front() = MatrixXd(numOfHidden.front(), numOfInput);
        biases.front()  = VectorXd(numOfHidden.front());
        deltas.front()  = VectorXd(numOfHidden.front());

        for (size_t i = 1; i < numOfHidden.size(); i++)
        {
            weights.at(i) = MatrixXd(numOfHidden.at(i), numOfHidden.at(i - 1));
            biases.at(i)  = VectorXd(numOfHidden.at(i));
            deltas.at(i)  = VectorXd(numOfHidden.at(i));
        }
        weights.at(numOfHidden.size()) =
            MatrixXd(numOfOutput, numOfHidden.back());
        biases.at(numOfHidden.size()) = VectorXd(numOfOutput);
        deltas.at(numOfHidden.size()) = VectorXd(numOfOutput);

        for (size_t i = 0; i < numOfHidden.size(); i++)
        {
            hidden.at(i) = VectorXd(numOfHidden.at(i));
        }
        for (MatrixXd& item : weights)
        {
            RandomizeMatrix(item);
        }
        for (VectorXd& item : biases)
        {
            RandomizeVector(item);
        }
    }
}

void BPNeuralNetwork::SetInput(const vector<double>& input)
{
    if (input.size() == this->input.rows())
    {
        int n = input.size();
        for (int i = 0; i < n; i++)
        {
            this->input(i) = input[i];
        }
    }
    else
    {
        throw string("input size does not match");
    }
}

const Eigen::VectorXd& BPNeuralNetwork::GetOutput() const
{
    return output;
}

double BPNeuralNetwork::Mse(const std::vector<double>& targets) const
{
    if (targets.size() == this->output.rows())
    {
        int n	  = targets.size();
        double sum = 0.0;
        for (int i = 0; i < n; i++)
        {
            double t = (targets[i] - output(i));
            sum		 = sum + t * t;
        }
        return sum / n;
    }
    else
    {
        throw string("targets size does not match");
    }
}

void BPNeuralNetwork::SetLearningRate(double learningRate)
{
    this->learningRate = learningRate;
}

void BPNeuralNetwork::FeedForward()
{
    Activte(hidden.front(), weights.front() * input + biases.front());
    for (size_t i = 1; i < hidden.size(); i++)
    {
        Activte(hidden.at(i), weights.at(i) * hidden.at(i - 1) + biases.at(i));
    }
    Activte(output, weights.back() * hidden.back() + biases.back());
}

void BPNeuralNetwork::Backpropagation(const Eigen::VectorXd& targets)
{
    BackpropagationToOutputLayer(targets);
    BackpropagationToHiddenLayers();
    UpdateWeights();
}

void BPNeuralNetwork::BackpropagationToOutputLayer(
    const Eigen::VectorXd& targets)
{
    VectorXd errorFactor(targets.size());
    for (int i = 0; i < errorFactor.size(); i++)
    {
        errorFactor(i) = -(targets(i) - output(i));
    }

    MatrixXd derivative = DerivativeActive(output);
    for (long i = 0; i < derivative.size(); i++)
    {
        deltas.back()(i) = derivative(i) * errorFactor(i);
    }
}

void BPNeuralNetwork::BackpropagationToHiddenLayers()
{
    for (int k = static_cast<int>(hidden.size() - 1); k >= 0; k--)
    {
        size_t uk = static_cast<size_t>(k);
        VectorXd errorFactor =
            (deltas.at(uk + 1).transpose() * weights.at(uk + 1)).transpose();
        VectorXd derivative = DerivativeActive(hidden.at(uk));
        for (long i = 0; i < derivative.size(); i++)
        {
            deltas.at(uk)(i) = derivative(i) * errorFactor(i);
        }
    }
}

void BPNeuralNetwork::UpdateWeights()
{
    UpdateFirstWeights();
    UpdateMiddleWeights();
    UpdateLastWeights();
}

void BPNeuralNetwork::UpdateLastWeights()
{
    biases.back()	= biases.back() - learningRate * deltas.back();
    MatrixXd& weight = weights.back();
    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) = weight(i, j) -
                           learningRate * hidden.back()(j) * deltas.back()(i);
        }
    }
}

void BPNeuralNetwork::UpdateMiddleWeights()
{
    for (int k = static_cast<int>(hidden.size() - 2); k >= 0; k--)
    {
        size_t uk = static_cast<size_t>(k);
        biases.at(uk + 1) =
            biases.at(uk + 1) - learningRate * deltas.at(uk + 1);
        MatrixXd& weight = weights.at(uk + 1);

        for (int i = 0; i < weight.rows(); i++)
        {
            for (int j = 0; j < weight.cols(); j++)
            {
                weight(i, j) = weight(i, j) - learningRate * hidden.at(uk)(j) *
                                                  deltas.at(uk + 1)(i);
            }
        }
    }
}

void BPNeuralNetwork::UpdateFirstWeights()
{
    biases.front()   = biases.front() - learningRate * deltas.front();
    MatrixXd& weight = weights.front();
    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) =
                weight(i, j) - learningRate * input(j) * deltas.front()(i);
        }
    }
}

void BPNeuralNetwork::RandomizeMatrix(MatrixXd& matrix)
{
    std::random_device rd;
    std::minstd_rand generator(rd());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);
    for (long i = 0; i < matrix.size(); i++)
    {
        matrix(i) = distribution(generator);
    }
}

void BPNeuralNetwork::RandomizeVector(Eigen::VectorXd& vec)
{
    std::random_device rd;
    std::minstd_rand generator(rd());
    std::uniform_real_distribution<> distribution(-1.0, 1.0);
    for (long i = 0; i < vec.size(); i++)
    {
        vec(i) = distribution(generator);
    }
}

void BPNeuralNetwork::Activte(VectorXd& result, const VectorXd& input)
{
    for (int i = 0; i < input.size(); i++)
    {
        result(i) = 1.0 / (1.0 + std::exp(-input(i)));
    }
}

Eigen::VectorXd BPNeuralNetwork::DerivativeActive(const VectorXd& input)
{
    VectorXd result(input.size());
    for (long i = 0; i < input.size(); i++)
    {
        result(i) = input(i) * (1.0 - input(i));
    }
    return result;
}
