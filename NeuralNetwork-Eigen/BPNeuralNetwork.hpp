#ifndef BPNEURALNETWORK_HPP
#define BPNEURALNETWORK_HPP
#include <eigen3/Eigen/Dense>
#include <vector>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

class BPNeuralNetwork
{
public:
    VectorXd input;
    vector<VectorXd> hidden;
    VectorXd output;

    vector<MatrixXd> weights;
    vector<VectorXd> biases;
    vector<VectorXd> deltas;

    double learningRate = 0.5;

    BPNeuralNetwork(int numOfInput, vector<int> numOfHidden, int numOfOutput);
    void FeedForward();
    void Backpropagation(const VectorXd& targets);

private:
    void BackpropagationToOutputLayer(const VectorXd& targets);
    void BackpropagationToHiddenLayers();
    void UpdateWeights();
    void UpdateLastWeights();
    void UpdateMiddleWeights();
    void UpdateFirstWeights();
    void RandomizeMatrix(MatrixXd& matrix);
    void RandomizeVector(VectorXd& vec);

    static VectorXd& First(vector<VectorXd>& vec);
    static MatrixXd& First(vector<MatrixXd>& vec);
    static VectorXd& Last(vector<VectorXd>& vec);
    static MatrixXd& Last(vector<MatrixXd>& vec);
    static int& Last(vector<int>& vec);
};

void Activte(Eigen::VectorXd& result, const Eigen::VectorXd& input);
Eigen::VectorXd DerivativeActive(const Eigen::VectorXd& input);
#endif // BPNEURALNETWORK_HPP
