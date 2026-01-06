#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Core>
#include <functional>
#include <atomic>
#include <memory>
#include <random>

class BiteSimulator;

struct OptimizationResult {
    Eigen::Matrix<float, 6, 1> bestParams;  // [rx, ry, rz, tx, ty, tz]
    double bestReward = -1e9;
    int iterations = 0;
    bool converged = false;
};

struct OptimizationConfig {
    int maxIterations = 100;
    float learningRate = 0.02f;
    float convergenceThreshold = 1e-5f;
    int populationSize = 40;
    int eliteCount = 8;
    float initialStd = 3.0f;
    float minStd = 0.1f;
};

using ProgressCallback = std::function<void(int iter, int total, double currentReward, double bestReward)>;

class BaseOptimizer {
public:
    explicit BaseOptimizer(BiteSimulator* simulator);
    virtual ~BaseOptimizer() = default;
    
    virtual OptimizationResult optimize(const OptimizationConfig& config,
                                        ProgressCallback callback = nullptr) = 0;
    
    void cancel() { m_cancelled.store(true); }
    bool isCancelled() const { return m_cancelled.load(); }
    void resetCancel() { m_cancelled.store(false); }

protected:
    double evaluateAt(const Eigen::Matrix<float, 6, 1>& params);
    Eigen::Matrix<float, 6, 1> getCurrentParams() const;
    void applyParams(const Eigen::Matrix<float, 6, 1>& params);
    
    BiteSimulator* m_simulator;
    std::atomic<bool> m_cancelled{false};
};

class GradientOptimizer : public BaseOptimizer {
public:
    explicit GradientOptimizer(BiteSimulator* simulator);
    
    OptimizationResult optimize(const OptimizationConfig& config,
                                ProgressCallback callback = nullptr) override;

private:
    Eigen::Matrix<float, 6, 1> computeGradient(const Eigen::Matrix<float, 6, 1>& params, float eps);
    
    // Adam: m=momentum, v=variance, t=timestep (must reset each run)
    Eigen::Matrix<float, 6, 1> m_m;
    Eigen::Matrix<float, 6, 1> m_v;
    int m_t = 0;
    static constexpr float BETA1 = 0.9f;
    static constexpr float BETA2 = 0.999f;
    static constexpr float EPSILON = 1e-8f;
    
    void resetAdamState();
    Eigen::Matrix<float, 6, 1> adamStep(const Eigen::Matrix<float, 6, 1>& gradient, float lr);
};

class CEMDirectOptimizer : public BaseOptimizer {
public:
    explicit CEMDirectOptimizer(BiteSimulator* simulator);
    
    OptimizationResult optimize(const OptimizationConfig& config,
                                ProgressCallback callback = nullptr) override;

private:
    std::mt19937 m_rng;
};

class MultiStartOptimizer : public BaseOptimizer {
public:
    explicit MultiStartOptimizer(BiteSimulator* simulator);
    
    OptimizationResult optimize(const OptimizationConfig& config,
                                ProgressCallback callback = nullptr) override;
    
    void setNumStarts(int n) { m_numStarts = n; }
    void setPerturbationScale(float rotDeg, float transMm) { 
        m_perturbRotDeg = rotDeg; 
        m_perturbTransMm = transMm;
    }

private:
    int m_numStarts = 5;
    float m_perturbRotDeg = 5.0f;
    float m_perturbTransMm = 2.0f;
    std::mt19937 m_rng;
    
    std::unique_ptr<GradientOptimizer> m_innerOptimizer;
};

#endif
