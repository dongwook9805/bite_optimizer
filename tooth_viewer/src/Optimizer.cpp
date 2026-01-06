#include "Optimizer.h"
#include "BiteSimulator.h"
#include <algorithm>
#include <cmath>
#include <iostream>

BaseOptimizer::BaseOptimizer(BiteSimulator* simulator)
    : m_simulator(simulator)
{
}

Eigen::Matrix<float, 6, 1> BaseOptimizer::getCurrentParams() const {
    Eigen::Matrix<float, 6, 1> params;
    params.head<3>() = m_simulator->currentRotation();
    params.tail<3>() = m_simulator->currentTranslation();
    return params;
}

void BaseOptimizer::applyParams(const Eigen::Matrix<float, 6, 1>& params) {
    m_simulator->reset();
    Eigen::Vector3f rot = params.head<3>();
    Eigen::Vector3f trans = params.tail<3>();
    m_simulator->applyTransform(rot, trans);
}

double BaseOptimizer::evaluateAt(const Eigen::Matrix<float, 6, 1>& params) {
    Eigen::Matrix<float, 6, 1> saved = getCurrentParams();
    applyParams(params);
    double reward = m_simulator->computeCachedReward();
    applyParams(saved);
    return reward;
}

GradientOptimizer::GradientOptimizer(BiteSimulator* simulator)
    : BaseOptimizer(simulator)
{
    resetAdamState();
}

void GradientOptimizer::resetAdamState() {
    m_m = Eigen::Matrix<float, 6, 1>::Zero();
    m_v = Eigen::Matrix<float, 6, 1>::Zero();
    m_t = 0;
}

Eigen::Matrix<float, 6, 1> GradientOptimizer::computeGradient(
    const Eigen::Matrix<float, 6, 1>& params, float eps) {
    
    Eigen::Matrix<float, 6, 1> grad;
    
    // Central difference: (f(x+h) - f(x-h)) / (2h)
    for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<float, 6, 1> paramsPlus = params;
        Eigen::Matrix<float, 6, 1> paramsMinus = params;
        paramsPlus(i) += eps;
        paramsMinus(i) -= eps;
        
        double rewardPlus = evaluateAt(paramsPlus);
        double rewardMinus = evaluateAt(paramsMinus);
        
        grad(i) = static_cast<float>((rewardPlus - rewardMinus) / (2.0f * eps));
    }
    
    return grad;
}

Eigen::Matrix<float, 6, 1> GradientOptimizer::adamStep(
    const Eigen::Matrix<float, 6, 1>& gradient, float lr) {
    
    m_t++;
    
    m_m = BETA1 * m_m + (1.0f - BETA1) * gradient;
    m_v = BETA2 * m_v + (1.0f - BETA2) * gradient.cwiseProduct(gradient);
    
    float biasCorrection1 = 1.0f - std::pow(BETA1, m_t);
    float biasCorrection2 = 1.0f - std::pow(BETA2, m_t);
    
    Eigen::Matrix<float, 6, 1> mHat = m_m / biasCorrection1;
    Eigen::Matrix<float, 6, 1> vHat = m_v / biasCorrection2;
    
    Eigen::Matrix<float, 6, 1> delta;
    for (int i = 0; i < 6; ++i) {
        delta(i) = lr * mHat(i) / (std::sqrt(vHat(i)) + EPSILON);
    }
    
    return delta;
}

OptimizationResult GradientOptimizer::optimize(
    const OptimizationConfig& config, ProgressCallback callback) {
    
    m_cancelled.store(false);
    resetAdamState();
    m_simulator->cacheSamplePoints();
    
    OptimizationResult result;
    result.bestParams = getCurrentParams();
    result.bestReward = evaluateAt(result.bestParams);
    
    Eigen::Matrix<float, 6, 1> currentParams = result.bestParams;
    double prevReward = result.bestReward;
    
    const float eps = 0.1f;
    
    for (int iter = 0; iter < config.maxIterations; ++iter) {
        if (m_cancelled.load()) break;
        
        Eigen::Matrix<float, 6, 1> grad = computeGradient(currentParams, eps);
        Eigen::Matrix<float, 6, 1> delta = adamStep(grad, config.learningRate);
        
        currentParams += delta;
        applyParams(currentParams);
        
        double currentReward = evaluateAt(currentParams);
        
        if (currentReward > result.bestReward) {
            result.bestReward = currentReward;
            result.bestParams = currentParams;
        }
        
        if (callback) {
            callback(iter + 1, config.maxIterations, currentReward, result.bestReward);
        }
        
        if (std::abs(currentReward - prevReward) < config.convergenceThreshold) {
            result.converged = true;
            result.iterations = iter + 1;
            break;
        }
        
        prevReward = currentReward;
        result.iterations = iter + 1;
    }
    
    applyParams(result.bestParams);
    return result;
}

CEMDirectOptimizer::CEMDirectOptimizer(BiteSimulator* simulator)
    : BaseOptimizer(simulator)
    , m_rng(std::random_device{}())
{
}

OptimizationResult CEMDirectOptimizer::optimize(
    const OptimizationConfig& config, ProgressCallback callback) {
    
    m_cancelled.store(false);
    m_simulator->cacheSamplePoints();
    
    OptimizationResult result;
    result.bestParams = getCurrentParams();
    result.bestReward = evaluateAt(result.bestParams);
    
    Eigen::Matrix<float, 6, 1> mean = result.bestParams;
    Eigen::Matrix<float, 6, 1> stddev;
    stddev << config.initialStd, config.initialStd, config.initialStd,
              config.initialStd, config.initialStd, config.initialStd;
    
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int gen = 0; gen < config.maxIterations; ++gen) {
        if (m_cancelled.load()) break;
        
        std::vector<std::pair<Eigen::Matrix<float, 6, 1>, double>> population;
        population.reserve(config.populationSize);
        
        for (int i = 0; i < config.populationSize; ++i) {
            if (m_cancelled.load()) break;
            
            Eigen::Matrix<float, 6, 1> candidate;
            for (int j = 0; j < 6; ++j) {
                candidate(j) = mean(j) + stddev(j) * dist(m_rng);
            }
            
            double reward = evaluateAt(candidate);
            population.emplace_back(candidate, reward);
            
            if (reward > result.bestReward) {
                result.bestReward = reward;
                result.bestParams = candidate;
            }
        }
        
        if (m_cancelled.load()) break;
        
        std::sort(population.begin(), population.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        Eigen::Matrix<float, 6, 1> newMean = Eigen::Matrix<float, 6, 1>::Zero();
        for (int i = 0; i < config.eliteCount; ++i) {
            newMean += population[i].first;
        }
        newMean /= config.eliteCount;
        
        Eigen::Matrix<float, 6, 1> newVar = Eigen::Matrix<float, 6, 1>::Zero();
        for (int i = 0; i < config.eliteCount; ++i) {
            Eigen::Matrix<float, 6, 1> diff = population[i].first - newMean;
            newVar += diff.cwiseProduct(diff);
        }
        newVar /= config.eliteCount;
        
        for (int j = 0; j < 6; ++j) {
            stddev(j) = std::max(config.minStd, std::sqrt(newVar(j)));
        }
        mean = newMean;
        
        double eliteMeanReward = 0.0;
        for (int i = 0; i < config.eliteCount; ++i) {
            eliteMeanReward += population[i].second;
        }
        eliteMeanReward /= config.eliteCount;
        
        if (callback) {
            callback(gen + 1, config.maxIterations, eliteMeanReward, result.bestReward);
        }
        
        result.iterations = gen + 1;
    }
    
    applyParams(result.bestParams);
    return result;
}

MultiStartOptimizer::MultiStartOptimizer(BiteSimulator* simulator)
    : BaseOptimizer(simulator)
    , m_rng(std::random_device{}())
{
    m_innerOptimizer = std::make_unique<GradientOptimizer>(simulator);
}

OptimizationResult MultiStartOptimizer::optimize(
    const OptimizationConfig& config, ProgressCallback callback) {
    
    m_cancelled.store(false);
    m_simulator->cacheSamplePoints();
    
    OptimizationResult globalBest;
    globalBest.bestParams = getCurrentParams();
    globalBest.bestReward = evaluateAt(globalBest.bestParams);
    
    Eigen::Matrix<float, 6, 1> initialParams = globalBest.bestParams;
    
    std::uniform_real_distribution<float> rotDist(-m_perturbRotDeg, m_perturbRotDeg);
    std::uniform_real_distribution<float> transDist(-m_perturbTransMm, m_perturbTransMm);
    
    int iterPerStart = config.maxIterations / m_numStarts;
    int totalIter = 0;
    
    for (int start = 0; start < m_numStarts; ++start) {
        if (m_cancelled.load()) break;
        
        Eigen::Matrix<float, 6, 1> startParams = initialParams;
        if (start > 0) {
            for (int j = 0; j < 3; ++j) {
                startParams(j) += rotDist(m_rng);
                startParams(j + 3) += transDist(m_rng);
            }
        }
        
        applyParams(startParams);
        
        OptimizationConfig localConfig = config;
        localConfig.maxIterations = iterPerStart;
        
        m_innerOptimizer->resetCancel();
        
        auto localCallback = [&](int iter, int total, double current, double best) {
            totalIter++;
            if (callback) {
                callback(totalIter, config.maxIterations, current, globalBest.bestReward);
            }
        };
        
        OptimizationResult localResult = m_innerOptimizer->optimize(localConfig, localCallback);
        
        if (localResult.bestReward > globalBest.bestReward) {
            globalBest.bestReward = localResult.bestReward;
            globalBest.bestParams = localResult.bestParams;
        }
        
        if (m_cancelled.load()) {
            m_innerOptimizer->cancel();
            break;
        }
    }
    
    globalBest.iterations = totalIter;
    applyParams(globalBest.bestParams);
    return globalBest;
}
