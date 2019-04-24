#ifndef BPNEURALNETWORK_HPP
#define BPNEURALNETWORK_HPP
#include <eigen3/Eigen/Dense>
#include <vector>

class BPNeuralNetwork
{
  private:
    typedef Eigen::MatrixXd MatrixXd;
    typedef Eigen::VectorXd VectorXd;

    VectorXd input;
    std::vector<VectorXd> hidden;
    VectorXd output;

    std::vector<MatrixXd> weights;
    std::vector<VectorXd> biases;
    std::vector<VectorXd> deltas;

    double learningRate = 0.5;

  public:
    BPNeuralNetwork(int numOfInput, std::vector<int> numOfHidden,
                    int numOfOutput);
    void SetInput(const std::vector<double>& input);
    const VectorXd& GetOutput() const;
    double Mse(const std::vector<double>& targets) const;
    void SetLearningRate(double learningRate);
    void FeedForward();
    void Backpropagation(const VectorXd& targets);

  private:
    void BackpropagationToOutputLayer(const VectorXd& targets);
    void BackpropagationToHiddenLayers();
    void UpdateWeights();
    void UpdateLastWeights();
    void UpdateMiddleWeights();
    void UpdateFirstWeights();
    static void RandomizeMatrix(MatrixXd& matrix);
    static void RandomizeVector(VectorXd& vec);

    static void Activte(Eigen::VectorXd& result, const Eigen::VectorXd& input);
    static Eigen::VectorXd DerivativeActive(const Eigen::VectorXd& input);
};

#endif // BPNEURALNETWORK_HPP
