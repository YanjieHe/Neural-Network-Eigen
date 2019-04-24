#include "Test.hpp"
#include "BPNeuralNetwork.hpp"
#include <ctime>
#include <iostream>
#include <vector>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
void Test1()
{
    BPNeuralNetwork network(3, {3, 3}, 1);
    vector<vector<double>> X = {{0, 0, 1}, {0, 1, 1}, {1, 0, 1}, {1, 1, 1}};
    vector<double> Y		 = {0, 1, 1, 0};
    for (int i = 0; i < 10000; i++)
    {
        for (size_t j = 0; j < X.size(); j++)
        {
            network.SetInput(X.at(j));
            VectorXd targets = VectorXd(1, 1);
            targets << Y.at(j);
            network.FeedForward();
            network.Backpropagation(targets);
        }
        if (i % 100 == 0)
        {
            cout << "output: " << endl;
            for (size_t k = 0; k < Y.size(); k++)
            {
                network.SetInput(X.at(k));
                network.FeedForward();
                cout << network.GetOutput() << endl;
            }
        }
    }
}

void Test2()
{
    vector<double> x1 = {0,		0.0625, 0.125, 0.1875, 0.25,  0.3125,
                         0.375, 0.4375, 0.5,   0.5625, 0.625, 0.6875,
                         0.75,  0.8125, 0.875, 0.9375, 1};
    vector<double> x2 = {
        0.06443691, 0.1229096,  0.18138229, 0.23985499, 0.29832768, 0.35680037,
        0.41527307, 0.47374576, 0.53221845, 0.59069115, 0,			0.70763653,
        0.76610923, 0.82458192, 0.88305461, 0.94152731, 1};
    vector<double> y = {
        0.70815181, 0.49598219, 0.31004938, 0.16109982, 0.05721754, 0.00314648,
        0,			0.04487366, 0.13137767, 0.2500242,  0.38919547, 0.53611192,
        0.677994,   0.80322393, 0.90231387, 0.96887404, 1};
    BPNeuralNetwork network(2, {3, 3}, 1);
    VectorXd targets(1);
    for (int i = 0; i < 20000; i++)
    {
        for (size_t j = 0; j < x1.size(); j++)
        {
            vector<double> input = {x1.at(j), x2.at(j)};
            network.SetInput(input);
            targets(0) = y.at(j);
            network.FeedForward();
            network.Backpropagation(targets);
        }
        cout << i << endl;
    }
    cout << "output: " << endl;
    for (size_t k = 0; k < y.size(); k++)
    {
        vector<double> input = {x1.at(k), x2.at(k)};
        network.SetInput(input);
        network.FeedForward();
        cout << network.GetOutput() << ", ";
    }
    cout << endl;
}

// void Test3()
//{
//    BPNeuralNetwork network(2, {2}, 2);
//    network.SetInput({0.05, 0.1});
//    network.weights.at(0) << 0.15, 0.20, 0.25, 0.30;
//    network.weights.at(1) << 0.40, 0.45, 0.50, 0.55;
//    network.biases.at(0) << 0.35, 0.35;
//    network.biases.at(1) << 0.60, 0.60;
//    VectorXd targets(2);
//    targets << 0.01, 0.99;
//    network.FeedForward();
//    network.Backpropagation(targets);
//    cout << "output: " << endl << network.output << endl;
//    cout << "weights 0:" << endl << network.weights.at(0) << endl;
//    cout << "weights 1:" << endl << network.weights.at(1) << endl;
//}

// void TimeIt()
//{
//    clock_t begin = clock();
//    Test1();
//    clock_t end			= clock();
//    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//    cout << "elapsed seconds: " << elapsed_secs << endl;
//}
