#include "RLOptimizer.h"
#include "BiteSimulator.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <fstream>
#include <iostream>

PolicyNetwork::PolicyNetwork(int input_dim, int hidden1, int hidden2, int output_dim) {
    m_W1 = Eigen::MatrixXf::Zero(hidden1, input_dim);
    m_b1 = Eigen::VectorXf::Zero(hidden1);
    m_W2 = Eigen::MatrixXf::Zero(hidden2, hidden1);
    m_b2 = Eigen::VectorXf::Zero(hidden2);
    m_W3 = Eigen::MatrixXf::Zero(output_dim, hidden2);
    m_b3 = Eigen::VectorXf::Zero(output_dim);
    
    randomize();
}

void PolicyNetwork::randomize(float scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    auto fillRandom = [&](Eigen::MatrixXf& mat) {
        for (int i = 0; i < mat.rows(); ++i)
            for (int j = 0; j < mat.cols(); ++j)
                mat(i, j) = dist(gen);
    };
    auto fillRandomVec = [&](Eigen::VectorXf& vec) {
        for (int i = 0; i < vec.size(); ++i)
            vec(i) = dist(gen);
    };
    
    fillRandom(m_W1);
    fillRandomVec(m_b1);
    fillRandom(m_W2);
    fillRandomVec(m_b2);
    fillRandom(m_W3);
    fillRandomVec(m_b3);
}

Eigen::VectorXf PolicyNetwork::tanh_activation(const Eigen::VectorXf& x) {
    return x.array().tanh().matrix();
}

Eigen::VectorXf PolicyNetwork::forward(const Eigen::VectorXf& input) const {
    Eigen::VectorXf h1 = tanh_activation(m_W1 * input + m_b1);
    Eigen::VectorXf h2 = tanh_activation(m_W2 * h1 + m_b2);
    return m_W3 * h2 + m_b3;
}

int PolicyNetwork::parameterCount() const {
    return m_W1.size() + m_b1.size() + m_W2.size() + m_b2.size() + m_W3.size() + m_b3.size();
}

Eigen::VectorXf PolicyNetwork::getParameters() const {
    Eigen::VectorXf params(parameterCount());
    int idx = 0;
    
    for (int i = 0; i < m_W1.size(); ++i) params(idx++) = m_W1.data()[i];
    for (int i = 0; i < m_b1.size(); ++i) params(idx++) = m_b1(i);
    for (int i = 0; i < m_W2.size(); ++i) params(idx++) = m_W2.data()[i];
    for (int i = 0; i < m_b2.size(); ++i) params(idx++) = m_b2(i);
    for (int i = 0; i < m_W3.size(); ++i) params(idx++) = m_W3.data()[i];
    for (int i = 0; i < m_b3.size(); ++i) params(idx++) = m_b3(i);
    
    return params;
}

void PolicyNetwork::setParameters(const Eigen::VectorXf& params) {
    int idx = 0;
    
    for (int i = 0; i < m_W1.size(); ++i) m_W1.data()[i] = params(idx++);
    for (int i = 0; i < m_b1.size(); ++i) m_b1(i) = params(idx++);
    for (int i = 0; i < m_W2.size(); ++i) m_W2.data()[i] = params(idx++);
    for (int i = 0; i < m_b2.size(); ++i) m_b2(i) = params(idx++);
    for (int i = 0; i < m_W3.size(); ++i) m_W3.data()[i] = params(idx++);
    for (int i = 0; i < m_b3.size(); ++i) m_b3(i) = params(idx++);
}

bool PolicyNetwork::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    
    auto writeMatrix = [&](const Eigen::MatrixXf& mat) {
        int rows = mat.rows(), cols = mat.cols();
        file.write(reinterpret_cast<const char*>(&rows), sizeof(int));
        file.write(reinterpret_cast<const char*>(&cols), sizeof(int));
        file.write(reinterpret_cast<const char*>(mat.data()), mat.size() * sizeof(float));
    };
    auto writeVector = [&](const Eigen::VectorXf& vec) {
        int size = vec.size();
        file.write(reinterpret_cast<const char*>(&size), sizeof(int));
        file.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(float));
    };
    
    writeMatrix(m_W1); writeVector(m_b1);
    writeMatrix(m_W2); writeVector(m_b2);
    writeMatrix(m_W3); writeVector(m_b3);
    
    return file.good();
}

bool PolicyNetwork::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    auto readMatrix = [&](Eigen::MatrixXf& mat) {
        int rows, cols;
        file.read(reinterpret_cast<char*>(&rows), sizeof(int));
        file.read(reinterpret_cast<char*>(&cols), sizeof(int));
        mat.resize(rows, cols);
        file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(float));
    };
    auto readVector = [&](Eigen::VectorXf& vec) {
        int size;
        file.read(reinterpret_cast<char*>(&size), sizeof(int));
        vec.resize(size);
        file.read(reinterpret_cast<char*>(vec.data()), size * sizeof(float));
    };
    
    readMatrix(m_W1); readVector(m_b1);
    readMatrix(m_W2); readVector(m_b2);
    readMatrix(m_W3); readVector(m_b3);
    
    return file.good();
}

CEMOptimizer::CEMOptimizer(BiteSimulator* simulator, const Config& config)
    : m_simulator(simulator)
    , m_config(config)
    , m_rng(std::random_device{}())
{
    if (config.mode == Mode::DIRECT) {
        m_mean = Eigen::VectorXf::Zero(6);
        m_std = Eigen::VectorXf::Constant(6, config.initial_std);
    } else {
        m_policy = std::make_unique<PolicyNetwork>();
        m_mean = m_policy->getParameters();
        m_std = Eigen::VectorXf::Constant(m_mean.size(), config.initial_std);
    }
    
    m_bestSolution = m_mean;
}

RLState CEMOptimizer::getCurrentState() const {
    RLState state;
    
    state.transform.head<3>() = m_simulator->currentRotation();
    state.transform.tail<3>() = m_simulator->currentTranslation();
    
    // Normalize: rotation [-30,30]deg -> [-1,1], translation [-10,10]mm -> [-1,1]
    state.transform.head<3>() /= 30.0f;
    state.transform.tail<3>() /= 10.0f;
    
    auto metrics = m_simulator->computeMetrics();
    state.protection_ratio = static_cast<float>(metrics.protection_ratio);
    state.balance_error = static_cast<float>(std::min(1.0, metrics.balance_error * 2.0));
    state.axial_score = static_cast<float>(metrics.axial_alignment_score);
    state.evenness = static_cast<float>(metrics.contact_evenness);
    state.contact_density = static_cast<float>(std::min(1.0, metrics.contact_point_count / 500.0));
    
    return state;
}

double CEMOptimizer::evaluate(const Eigen::VectorXf& candidate) {
    if (m_config.mode == Mode::DIRECT) {
        Eigen::Vector3f savedRot = m_simulator->currentRotation();
        Eigen::Vector3f savedTrans = m_simulator->currentTranslation();
        
        m_simulator->reset();
        Eigen::Vector3f rot = candidate.head<3>();
        Eigen::Vector3f trans = candidate.tail<3>();
        m_simulator->applyTransform(rot, trans);
        
        auto metrics = m_simulator->computeMetrics();
        double reward = m_simulator->computeReward(metrics);
        
        m_simulator->reset();
        m_simulator->applyTransform(savedRot, savedTrans);
        
        return reward;
    } else {
        PolicyNetwork tempPolicy = *m_policy;
        tempPolicy.setParameters(candidate);
        return runPolicyEpisode(tempPolicy, m_config.episode_steps);
    }
}

double CEMOptimizer::runPolicyEpisode(const PolicyNetwork& policy, int steps) {
    Eigen::Vector3f savedRot = m_simulator->currentRotation();
    Eigen::Vector3f savedTrans = m_simulator->currentTranslation();
    
    m_simulator->reset();
    double totalReward = 0.0;
    
    for (int step = 0; step < steps; ++step) {
        RLState state = getCurrentState();
        Eigen::VectorXf action = policy.forward(state.toVector());
        
        Eigen::Vector3f deltaRot = action.head<3>() * m_config.action_scale;
        Eigen::Vector3f deltaTrans = action.tail<3>() * m_config.action_scale;
        m_simulator->applyTransform(deltaRot, deltaTrans);
        
        auto metrics = m_simulator->computeMetrics();
        totalReward += m_simulator->computeReward(metrics);
    }
    
    m_simulator->reset();
    m_simulator->applyTransform(savedRot, savedTrans);
    
    return totalReward / steps;
}

double CEMOptimizer::optimize(std::function<void(int, double, double)> progressCallback) {
    m_cancelRequested.store(false);
    
    std::normal_distribution<float> normalDist(0.0f, 1.0f);
    
    for (int gen = 0; gen < m_config.max_generations; ++gen) {
        if (m_cancelRequested.load()) break;
        
        std::vector<std::pair<Eigen::VectorXf, double>> candidates(m_config.population_size);
        
        for (int i = 0; i < m_config.population_size; ++i) {
            if (m_cancelRequested.load()) break;
            
            Eigen::VectorXf noise(m_mean.size());
            for (int j = 0; j < noise.size(); ++j) {
                noise(j) = normalDist(m_rng);
            }
            
            candidates[i].first = m_mean + m_std.cwiseProduct(noise);
            candidates[i].second = evaluate(candidates[i].first);
            
            if (candidates[i].second > m_bestReward) {
                m_bestReward = candidates[i].second;
                m_bestSolution = candidates[i].first;
            }
        }
        
        if (m_cancelRequested.load()) break;
        
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        std::vector<Eigen::VectorXf> elites(m_config.elite_count);
        for (int i = 0; i < m_config.elite_count; ++i) {
            elites[i] = candidates[i].first;
        }
        
        m_mean = Eigen::VectorXf::Zero(m_mean.size());
        for (const auto& elite : elites) {
            m_mean += elite;
        }
        m_mean /= m_config.elite_count;
        
        m_std = Eigen::VectorXf::Zero(m_std.size());
        for (const auto& elite : elites) {
            Eigen::VectorXf diff = elite - m_mean;
            m_std += diff.cwiseProduct(diff);
        }
        m_std = (m_std / m_config.elite_count).cwiseSqrt();
        m_std = m_std.cwiseMax(m_config.min_std);
        
        if (m_config.std_decay < 1.0f) {
            m_std *= m_config.std_decay;
            m_std = m_std.cwiseMax(m_config.min_std);
        }
        
        if (progressCallback) {
            double meanReward = 0.0;
            for (int i = 0; i < m_config.elite_count; ++i) {
                meanReward += candidates[i].second;
            }
            meanReward /= m_config.elite_count;
            progressCallback(gen, m_bestReward, meanReward);
        }
    }
    
    if (m_config.mode == Mode::DIRECT) {
        m_simulator->reset();
        Eigen::Vector3f rot = m_bestSolution.head<3>();
        Eigen::Vector3f trans = m_bestSolution.tail<3>();
        m_simulator->applyTransform(rot, trans);
    } else {
        m_policy->setParameters(m_bestSolution);
    }
    
    return m_bestReward;
}

ESOptimizer::ESOptimizer(BiteSimulator* simulator, const Config& config)
    : m_simulator(simulator)
    , m_config(config)
    , m_rng(std::random_device{}())
{
    m_policy = std::make_unique<PolicyNetwork>();
}

ActorCriticNetwork::ActorCriticNetwork(int state_dim, int action_dim, int hidden) {
    m_actorW1 = Eigen::MatrixXf::Zero(hidden, state_dim);
    m_actorB1 = Eigen::VectorXf::Zero(hidden);
    m_actorW2 = Eigen::MatrixXf::Zero(action_dim, hidden);
    m_actorB2 = Eigen::VectorXf::Zero(action_dim);
    
    m_criticW1 = Eigen::MatrixXf::Zero(hidden, state_dim);
    m_criticB1 = Eigen::VectorXf::Zero(hidden);
    m_criticW2 = Eigen::MatrixXf::Zero(1, hidden);
    m_criticB2 = Eigen::VectorXf::Zero(1);
    
    m_logStd = Eigen::VectorXf::Constant(action_dim, -0.5f);
    
    randomize();
}

void ActorCriticNetwork::randomize(float scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    auto fill = [&](Eigen::MatrixXf& m) {
        for (int i = 0; i < m.size(); ++i) m.data()[i] = dist(gen);
    };
    auto fillV = [&](Eigen::VectorXf& v) {
        for (int i = 0; i < v.size(); ++i) v(i) = dist(gen);
    };
    
    fill(m_actorW1); fillV(m_actorB1);
    fill(m_actorW2); fillV(m_actorB2);
    fill(m_criticW1); fillV(m_criticB1);
    fill(m_criticW2); fillV(m_criticB2);
}

Eigen::VectorXf ActorCriticNetwork::tanh_act(const Eigen::VectorXf& x) {
    return x.array().tanh().matrix();
}

Eigen::VectorXf ActorCriticNetwork::actorForward(const Eigen::VectorXf& state) const {
    Eigen::VectorXf h = tanh_act(m_actorW1 * state + m_actorB1);
    return m_actorW2 * h + m_actorB2;
}

float ActorCriticNetwork::criticForward(const Eigen::VectorXf& state) const {
    Eigen::VectorXf h = tanh_act(m_criticW1 * state + m_criticB1);
    return (m_criticW2 * h + m_criticB2)(0);
}

int ActorCriticNetwork::actorParamCount() const {
    return m_actorW1.size() + m_actorB1.size() + m_actorW2.size() + m_actorB2.size() + m_logStd.size();
}

int ActorCriticNetwork::criticParamCount() const {
    return m_criticW1.size() + m_criticB1.size() + m_criticW2.size() + m_criticB2.size();
}

Eigen::VectorXf ActorCriticNetwork::getActorParams() const {
    Eigen::VectorXf p(actorParamCount());
    int idx = 0;
    for (int i = 0; i < m_actorW1.size(); ++i) p(idx++) = m_actorW1.data()[i];
    for (int i = 0; i < m_actorB1.size(); ++i) p(idx++) = m_actorB1(i);
    for (int i = 0; i < m_actorW2.size(); ++i) p(idx++) = m_actorW2.data()[i];
    for (int i = 0; i < m_actorB2.size(); ++i) p(idx++) = m_actorB2(i);
    for (int i = 0; i < m_logStd.size(); ++i) p(idx++) = m_logStd(i);
    return p;
}

Eigen::VectorXf ActorCriticNetwork::getCriticParams() const {
    Eigen::VectorXf p(criticParamCount());
    int idx = 0;
    for (int i = 0; i < m_criticW1.size(); ++i) p(idx++) = m_criticW1.data()[i];
    for (int i = 0; i < m_criticB1.size(); ++i) p(idx++) = m_criticB1(i);
    for (int i = 0; i < m_criticW2.size(); ++i) p(idx++) = m_criticW2.data()[i];
    for (int i = 0; i < m_criticB2.size(); ++i) p(idx++) = m_criticB2(i);
    return p;
}

void ActorCriticNetwork::setActorParams(const Eigen::VectorXf& p) {
    int idx = 0;
    for (int i = 0; i < m_actorW1.size(); ++i) m_actorW1.data()[i] = p(idx++);
    for (int i = 0; i < m_actorB1.size(); ++i) m_actorB1(i) = p(idx++);
    for (int i = 0; i < m_actorW2.size(); ++i) m_actorW2.data()[i] = p(idx++);
    for (int i = 0; i < m_actorB2.size(); ++i) m_actorB2(i) = p(idx++);
    for (int i = 0; i < m_logStd.size(); ++i) m_logStd(i) = p(idx++);
}

void ActorCriticNetwork::setCriticParams(const Eigen::VectorXf& p) {
    int idx = 0;
    for (int i = 0; i < m_criticW1.size(); ++i) m_criticW1.data()[i] = p(idx++);
    for (int i = 0; i < m_criticB1.size(); ++i) m_criticB1(i) = p(idx++);
    for (int i = 0; i < m_criticW2.size(); ++i) m_criticW2.data()[i] = p(idx++);
    for (int i = 0; i < m_criticB2.size(); ++i) m_criticB2(i) = p(idx++);
}

PPOOptimizer::PPOOptimizer(BiteSimulator* simulator, const Config& config)
    : m_simulator(simulator)
    , m_config(config)
    , m_rng(std::random_device{}())
{
    m_network = std::make_unique<ActorCriticNetwork>();
}

RLState PPOOptimizer::getCurrentState() const {
    RLState state;
    state.transform.head<3>() = m_simulator->currentRotation();
    state.transform.tail<3>() = m_simulator->currentTranslation();
    state.transform.head<3>() /= 30.0f;
    state.transform.tail<3>() /= 10.0f;
    
    auto metrics = m_simulator->computeMetrics();
    state.protection_ratio = static_cast<float>(metrics.protection_ratio);
    state.balance_error = static_cast<float>(std::min(1.0, metrics.balance_error * 2.0));
    state.axial_score = static_cast<float>(metrics.axial_alignment_score);
    state.evenness = static_cast<float>(metrics.contact_evenness);
    state.contact_density = static_cast<float>(std::min(1.0, metrics.contact_point_count / 500.0));
    return state;
}

Eigen::VectorXf PPOOptimizer::sampleAction(const Eigen::VectorXf& mean, float& logProb) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    Eigen::VectorXf action(mean.size());
    logProb = 0.0f;
    
    float logStd = -0.5f;
    float std = std::exp(logStd);
    
    for (int i = 0; i < mean.size(); ++i) {
        float noise = dist(m_rng);
        action(i) = mean(i) + std * noise;
        // log_prob = -0.5 * ((a - mu) / std)^2 - log(std) - 0.5 * log(2*pi)
        float diff = (action(i) - mean(i)) / std;
        logProb += -0.5f * diff * diff - logStd - 0.9189f;
    }
    return action;
}

float PPOOptimizer::computeLogProb(const Eigen::VectorXf& action, const Eigen::VectorXf& mean) {
    float logStd = -0.5f;
    float std = std::exp(logStd);
    float logProb = 0.0f;
    
    for (int i = 0; i < action.size(); ++i) {
        float diff = (action(i) - mean(i)) / std;
        logProb += -0.5f * diff * diff - logStd - 0.9189f;
    }
    return logProb;
}

void PPOOptimizer::collectTrajectory(std::vector<Transition>& buffer) {
    buffer.clear();
    m_simulator->reset();
    
    for (int step = 0; step < m_config.steps_per_iteration; ++step) {
        if (m_cancelRequested.load()) break;
        
        RLState state = getCurrentState();
        Eigen::VectorXf stateVec = state.toVector();
        
        Eigen::VectorXf actionMean = m_network->actorForward(stateVec);
        float logProb;
        Eigen::VectorXf action = sampleAction(actionMean, logProb);
        float value = m_network->criticForward(stateVec);
        
        Eigen::Vector3f deltaRot = action.head<3>() * m_config.action_scale;
        Eigen::Vector3f deltaTrans = action.tail<3>() * m_config.action_scale;
        m_simulator->applyTransform(deltaRot, deltaTrans);
        
        auto metrics = m_simulator->computeMetrics();
        float reward = static_cast<float>(m_simulator->computeReward(metrics));
        
        RLState nextState = getCurrentState();
        
        Transition t;
        t.state = stateVec;
        t.action = action;
        t.reward = reward;
        t.next_state = nextState.toVector();
        t.done = false;
        t.log_prob = logProb;
        t.value = value;
        buffer.push_back(t);
    }
}

void PPOOptimizer::computeGAE(std::vector<Transition>& buffer,
                              std::vector<float>& advantages,
                              std::vector<float>& returns) {
    int n = buffer.size();
    advantages.resize(n);
    returns.resize(n);
    
    float lastGae = 0.0f;
    float lastValue = buffer.empty() ? 0.0f : m_network->criticForward(buffer.back().next_state);
    
    // GAE: A_t = delta_t + gamma * lambda * A_{t+1}
    // delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    for (int t = n - 1; t >= 0; --t) {
        float nextValue = (t == n - 1) ? lastValue : buffer[t + 1].value;
        float delta = buffer[t].reward + m_config.gamma * nextValue - buffer[t].value;
        lastGae = delta + m_config.gamma * m_config.gae_lambda * lastGae;
        advantages[t] = lastGae;
        returns[t] = advantages[t] + buffer[t].value;
    }
    
    // Normalize advantages
    float mean = 0.0f, var = 0.0f;
    for (float a : advantages) mean += a;
    mean /= n;
    for (float a : advantages) var += (a - mean) * (a - mean);
    var = std::sqrt(var / n + 1e-8f);
    for (float& a : advantages) a = (a - mean) / var;
}

void PPOOptimizer::updatePolicy(const std::vector<Transition>& buffer,
                                const std::vector<float>& advantages,
                                const std::vector<float>& returns) {
    int n = buffer.size();
    if (n == 0) return;
    
    Eigen::VectorXf actorParams = m_network->getActorParams();
    Eigen::VectorXf criticParams = m_network->getCriticParams();
    
    for (int epoch = 0; epoch < m_config.ppo_epochs; ++epoch) {
        if (m_cancelRequested.load()) break;
        
        // Simple full-batch gradient estimation via finite differences
        Eigen::VectorXf actorGrad = Eigen::VectorXf::Zero(actorParams.size());
        Eigen::VectorXf criticGrad = Eigen::VectorXf::Zero(criticParams.size());
        
        float totalPolicyLoss = 0.0f;
        float totalValueLoss = 0.0f;
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXf newMean = m_network->actorForward(buffer[i].state);
            float newLogProb = computeLogProb(buffer[i].action, newMean);
            float ratio = std::exp(newLogProb - buffer[i].log_prob);
            
            // Clipped surrogate objective
            float surr1 = ratio * advantages[i];
            float surr2 = std::clamp(ratio, 1.0f - m_config.clip_epsilon, 1.0f + m_config.clip_epsilon) * advantages[i];
            float policyLoss = -std::min(surr1, surr2);
            totalPolicyLoss += policyLoss;
            
            // Value loss
            float value = m_network->criticForward(buffer[i].state);
            float valueLoss = 0.5f * (returns[i] - value) * (returns[i] - value);
            totalValueLoss += valueLoss;
        }
        
        // Numerical gradient for actor (simplified)
        float eps = 1e-4f;
        for (int p = 0; p < std::min(100, (int)actorParams.size()); ++p) {
            Eigen::VectorXf paramsPlus = actorParams;
            Eigen::VectorXf paramsMinus = actorParams;
            paramsPlus(p) += eps;
            paramsMinus(p) -= eps;
            
            m_network->setActorParams(paramsPlus);
            float lossPlus = 0.0f;
            for (int i = 0; i < n; ++i) {
                Eigen::VectorXf mean = m_network->actorForward(buffer[i].state);
                float logP = computeLogProb(buffer[i].action, mean);
                float ratio = std::exp(logP - buffer[i].log_prob);
                float s1 = ratio * advantages[i];
                float s2 = std::clamp(ratio, 1.0f - m_config.clip_epsilon, 1.0f + m_config.clip_epsilon) * advantages[i];
                lossPlus -= std::min(s1, s2);
            }
            
            m_network->setActorParams(paramsMinus);
            float lossMinus = 0.0f;
            for (int i = 0; i < n; ++i) {
                Eigen::VectorXf mean = m_network->actorForward(buffer[i].state);
                float logP = computeLogProb(buffer[i].action, mean);
                float ratio = std::exp(logP - buffer[i].log_prob);
                float s1 = ratio * advantages[i];
                float s2 = std::clamp(ratio, 1.0f - m_config.clip_epsilon, 1.0f + m_config.clip_epsilon) * advantages[i];
                lossMinus -= std::min(s1, s2);
            }
            
            actorGrad(p) = (lossPlus - lossMinus) / (2.0f * eps);
        }
        m_network->setActorParams(actorParams);
        
        // Numerical gradient for critic
        for (int p = 0; p < std::min(100, (int)criticParams.size()); ++p) {
            Eigen::VectorXf paramsPlus = criticParams;
            Eigen::VectorXf paramsMinus = criticParams;
            paramsPlus(p) += eps;
            paramsMinus(p) -= eps;
            
            m_network->setCriticParams(paramsPlus);
            float lossPlus = 0.0f;
            for (int i = 0; i < n; ++i) {
                float v = m_network->criticForward(buffer[i].state);
                lossPlus += 0.5f * (returns[i] - v) * (returns[i] - v);
            }
            
            m_network->setCriticParams(paramsMinus);
            float lossMinus = 0.0f;
            for (int i = 0; i < n; ++i) {
                float v = m_network->criticForward(buffer[i].state);
                lossMinus += 0.5f * (returns[i] - v) * (returns[i] - v);
            }
            
            criticGrad(p) = (lossPlus - lossMinus) / (2.0f * eps);
        }
        m_network->setCriticParams(criticParams);
        
        // Gradient descent
        actorParams -= m_config.actor_lr * actorGrad;
        criticParams -= m_config.critic_lr * criticGrad;
        
        m_network->setActorParams(actorParams);
        m_network->setCriticParams(criticParams);
    }
}

double PPOOptimizer::optimize(std::function<void(int, double, double)> progressCallback) {
    m_cancelRequested.store(false);
    
    Eigen::Vector3f savedRot = m_simulator->currentRotation();
    Eigen::Vector3f savedTrans = m_simulator->currentTranslation();
    
    for (int iter = 0; iter < m_config.max_iterations; ++iter) {
        if (m_cancelRequested.load()) break;
        
        std::vector<Transition> buffer;
        collectTrajectory(buffer);
        
        if (buffer.empty()) break;
        
        std::vector<float> advantages, returns;
        computeGAE(buffer, advantages, returns);
        updatePolicy(buffer, advantages, returns);
        
        float meanReward = 0.0f;
        for (const auto& t : buffer) meanReward += t.reward;
        meanReward /= buffer.size();
        
        if (meanReward > m_bestReward) {
            m_bestReward = meanReward;
        }
        
        if (progressCallback) {
            progressCallback(iter, m_bestReward, meanReward);
        }
        
        m_simulator->reset();
        m_simulator->applyTransform(savedRot, savedTrans);
    }
    
    return m_bestReward;
}


RLState ESOptimizer::getCurrentState() const {
    RLState state;
    
    state.transform.head<3>() = m_simulator->currentRotation();
    state.transform.tail<3>() = m_simulator->currentTranslation();
    
    state.transform.head<3>() /= 30.0f;
    state.transform.tail<3>() /= 10.0f;
    
    auto metrics = m_simulator->computeMetrics();
    state.protection_ratio = static_cast<float>(metrics.protection_ratio);
    state.balance_error = static_cast<float>(std::min(1.0, metrics.balance_error * 2.0));
    state.axial_score = static_cast<float>(metrics.axial_alignment_score);
    state.evenness = static_cast<float>(metrics.contact_evenness);
    state.contact_density = static_cast<float>(std::min(1.0, metrics.contact_point_count / 500.0));
    
    return state;
}

double ESOptimizer::evaluatePolicy(const PolicyNetwork& policy, int steps) {
    Eigen::Vector3f savedRot = m_simulator->currentRotation();
    Eigen::Vector3f savedTrans = m_simulator->currentTranslation();
    
    m_simulator->reset();
    double totalReward = 0.0;
    
    for (int step = 0; step < steps; ++step) {
        RLState state = getCurrentState();
        Eigen::VectorXf action = policy.forward(state.toVector());
        
        Eigen::Vector3f deltaRot = action.head<3>() * m_config.action_scale;
        Eigen::Vector3f deltaTrans = action.tail<3>() * m_config.action_scale;
        m_simulator->applyTransform(deltaRot, deltaTrans);
        
        auto metrics = m_simulator->computeMetrics();
        totalReward += m_simulator->computeReward(metrics);
    }
    
    m_simulator->reset();
    m_simulator->applyTransform(savedRot, savedTrans);
    
    return totalReward / steps;
}

double ESOptimizer::optimize(std::function<void(int, double, double)> progressCallback) {
    m_cancelRequested.store(false);
    
    std::normal_distribution<float> normalDist(0.0f, 1.0f);
    
    Eigen::VectorXf params = m_policy->getParameters();
    int numParams = params.size();
    
    for (int gen = 0; gen < m_config.max_generations; ++gen) {
        if (m_cancelRequested.load()) break;
        
        std::vector<Eigen::VectorXf> noises(m_config.population_size);
        std::vector<double> rewards(m_config.population_size);
        
        for (int i = 0; i < m_config.population_size; ++i) {
            if (m_cancelRequested.load()) break;
            
            noises[i] = Eigen::VectorXf(numParams);
            for (int j = 0; j < numParams; ++j) {
                noises[i](j) = normalDist(m_rng);
            }
            
            PolicyNetwork perturbedPolicy = *m_policy;
            Eigen::VectorXf perturbedParams = params + m_config.noise_std * noises[i];
            perturbedPolicy.setParameters(perturbedParams);
            rewards[i] = evaluatePolicy(perturbedPolicy, m_config.episode_steps);
            
            if (rewards[i] > m_bestReward) {
                m_bestReward = rewards[i];
            }
        }
        
        if (m_cancelRequested.load()) break;
        
        // Fitness shaping: (r - mean) / std
        double meanReward = 0.0, stdReward = 0.0;
        for (double r : rewards) meanReward += r;
        meanReward /= rewards.size();
        
        for (double r : rewards) stdReward += (r - meanReward) * (r - meanReward);
        stdReward = std::sqrt(stdReward / rewards.size()) + 1e-8;
        
        std::vector<double> normalizedRewards(rewards.size());
        for (size_t i = 0; i < rewards.size(); ++i) {
            normalizedRewards[i] = (rewards[i] - meanReward) / stdReward;
        }
        
        // ES gradient: sum(normalized_reward * noise) / (pop_size * noise_std)
        Eigen::VectorXf gradient = Eigen::VectorXf::Zero(numParams);
        for (int i = 0; i < m_config.population_size; ++i) {
            gradient += normalizedRewards[i] * noises[i];
        }
        gradient /= (m_config.population_size * m_config.noise_std);
        
        params += m_config.learning_rate * gradient;
        m_policy->setParameters(params);
        
        if (progressCallback) {
            progressCallback(gen, m_bestReward, meanReward);
        }
    }
    
    return m_bestReward;
}
