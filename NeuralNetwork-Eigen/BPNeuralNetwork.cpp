#include "BPNeuralNetwork.hpp"
#include <cmath>
#include <iostream>
#include <random>
#include <string>
using namespace std;

BPNeuralNetwork::BPNeuralNetwork(int numOfInput, vector<int> numOfHidden,
                                 int numOfOutput)
    : input(numOfInput)
    , hidden(numOfHidden.size())
    , output(numOfOutput)
    , weights(numOfHidden.size() + 1)
    , biases(numOfHidden.size() + 1)
    , deltas(numOfHidden.size() + 1)
{
    if (numOfHidden.size() < 1)
    {
        throw string("There should be at least one hidden layer");
    }
    else
    {
        weights.at(0) = MatrixXd(numOfHidden.at(0), numOfInput);
        biases.at(0) = VectorXd(numOfHidden.at(0));
        deltas.at(0) = VectorXd(numOfHidden.at(0));

        for (size_t i = 1; i < numOfHidden.size(); i++)
        {
            weights.at(i) = MatrixXd(numOfHidden.at(i), numOfHidden.at(i - 1));
            biases.at(i) = VectorXd(numOfHidden.at(i));
            deltas.at(i) = VectorXd(numOfHidden.at(i));
        }
        weights.at(numOfHidden.size()) =
            MatrixXd(numOfOutput, Last(numOfHidden));
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

void BPNeuralNetwork::FeedForward()
{
    Activte(First(hidden), First(weights) * input + First(biases));
    for (size_t i = 1; i < hidden.size(); i++)
    {
        Activte(hidden.at(i), weights.at(i) * hidden.at(i - 1) + biases.at(i));
    }
    Activte(output, Last(weights) * Last(hidden) + Last(biases));
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
        Last(deltas)(i) = derivative(i) * errorFactor(i);
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
    Last(biases) = Last(biases) - learningRate * Last(deltas);
    MatrixXd& weight = Last(weights);
    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) =
                weight(i, j) - learningRate * Last(hidden)(j) * Last(deltas)(i);
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
    First(biases) = First(biases) - learningRate * First(deltas);
    MatrixXd& weight = First(weights);
    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) =
                weight(i, j) - learningRate * input(j) * First(deltas)(i);
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

Eigen::VectorXd& BPNeuralNetwork::First(vector<Eigen::VectorXd>& vec)
{
    return vec.at(0);
}

Eigen::MatrixXd& BPNeuralNetwork::First(vector<Eigen::MatrixXd>& vec)
{
    return vec.at(0);
}

Eigen::VectorXd& BPNeuralNetwork::Last(vector<Eigen::VectorXd>& vec)
{
    return vec.at(vec.size() - 1);
}

Eigen::MatrixXd& BPNeuralNetwork::Last(vector<Eigen::MatrixXd>& vec)
{
    return vec.at(vec.size() - 1);
}

int& BPNeuralNetwork::Last(vector<int>& vec)
{
    return vec.at(vec.size() - 1);
}

void Activte(VectorXd& result, const VectorXd& input)
{
    for (int i = 0; i < input.size(); i++)
    {
        result(i) = 1.0 / (1.0 + std::exp(-input(i)));
    }
}

VectorXd DerivativeActive(const VectorXd& input)
{
    VectorXd result(input.size());
    for (long i = 0; i < input.size(); i++)
    {
        result(i) = input(i) * (1.0 - input(i));
    }
    return result;
}
