#ifndef RLOPTIMIZER_H
#define RLOPTIMIZER_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <functional>
#include <random>
#include <atomic>
#include <memory>

class BiteSimulator;

/**
 * @brief State representation for RL optimization
 * 
 * State includes:
 * - Current 6-DOF transformation (normalized)
 * - Key orthodontic metrics (normalized to 0-1)
 */
struct RLState {
    // 6-DOF transformation (normalized)
    Eigen::Matrix<float, 6, 1> transform;  // [rx, ry, rz, tx, ty, tz]
    
    // Key metrics (normalized 0-1)
    float protection_ratio = 0.0f;    // Posterior force ratio
    float balance_error = 0.0f;       // L/R imbalance
    float axial_score = 0.0f;         // Vertical loading alignment
    float evenness = 0.0f;            // Contact distribution
    float contact_density = 0.0f;     // Contact point count (normalized)
    
    // Flatten to vector for neural network input
    Eigen::VectorXf toVector() const {
        Eigen::VectorXf v(11);
        v.segment<6>(0) = transform;
        v(6) = protection_ratio;
        v(7) = balance_error;
        v(8) = axial_score;
        v(9) = evenness;
        v(10) = contact_density;
        return v;
    }
    
    static constexpr int dimension() { return 11; }
};

/**
 * @brief Action representation for RL optimization
 * 
 * Action is a 6-DOF delta transformation
 */
struct RLAction {
    Eigen::Matrix<float, 6, 1> delta;  // [drx, dry, drz, dtx, dty, dtz]
    
    static constexpr int dimension() { return 6; }
};

/**
 * @brief Simple MLP (Multi-Layer Perceptron) for policy network
 * 
 * Architecture: State(11) -> Hidden(64) -> Hidden(32) -> Action(6)
 * Uses tanh activation for hidden layers
 * Output is unbounded (scaled by action_scale)
 */
class PolicyNetwork {
public:
    PolicyNetwork(int input_dim = RLState::dimension(), 
                  int hidden1 = 64, 
                  int hidden2 = 32,
                  int output_dim = RLAction::dimension());
    
    // Forward pass
    Eigen::VectorXf forward(const Eigen::VectorXf& input) const;
    
    // Get/set parameters as flat vector (for CEM, evolution strategies)
    Eigen::VectorXf getParameters() const;
    void setParameters(const Eigen::VectorXf& params);
    int parameterCount() const;
    
    // Initialize with random weights
    void randomize(float scale = 0.1f);
    
    // Save/load weights
    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    Eigen::MatrixXf m_W1, m_W2, m_W3;
    Eigen::VectorXf m_b1, m_b2, m_b3;
    
    // Activation function
    static Eigen::VectorXf tanh_activation(const Eigen::VectorXf& x);
};

/**
 * @brief Cross-Entropy Method (CEM) optimizer
 * 
 * Algorithm:
 * 1. Sample N candidates from Gaussian distribution
 * 2. Evaluate each candidate (run episode, get total reward)
 * 3. Select top K elites
 * 4. Update mean/std to fit elites
 * 5. Repeat
 * 
 * Can optimize:
 * - Direct 6-DOF transformation (mode: DIRECT)
 * - Neural network policy parameters (mode: POLICY)
 */
struct CEMConfig {
    enum class Mode { DIRECT, POLICY };
    Mode mode = Mode::DIRECT;
    int population_size = 50;
    int elite_count = 10;
    int max_generations = 100;
    float initial_std = 0.5f;
    float min_std = 0.01f;
    float std_decay = 0.95f;
    int episode_steps = 50;
    float action_scale = 0.1f;
};

class CEMOptimizer {
public:
    using Mode = CEMConfig::Mode;
    using Config = CEMConfig;
    
    explicit CEMOptimizer(BiteSimulator* simulator, const Config& config = Config{});
    
    // Run optimization
    // Returns best reward achieved
    // progressCallback(generation, best_reward, mean_reward)
    double optimize(std::function<void(int, double, double)> progressCallback = nullptr);
    
    // Get best solution found
    Eigen::VectorXf getBestSolution() const { return m_bestSolution; }
    double getBestReward() const { return m_bestReward; }
    
    // Get/set policy network (for POLICY mode)
    PolicyNetwork* policy() { return m_policy.get(); }
    const PolicyNetwork* policy() const { return m_policy.get(); }
    
    // Cancellation
    void cancel() { m_cancelRequested.store(true); }
    bool isCancelled() const { return m_cancelRequested.load(); }
    void resetCancellation() { m_cancelRequested.store(false); }

private:
    // Evaluate a candidate solution
    // For DIRECT mode: candidate is 6-DOF transform
    // For POLICY mode: candidate is policy parameters
    double evaluate(const Eigen::VectorXf& candidate);
    
    // Run one episode with policy network
    double runPolicyEpisode(const PolicyNetwork& policy, int steps);
    
    // Get current state from simulator
    RLState getCurrentState() const;
    
    BiteSimulator* m_simulator;
    Config m_config;
    
    // CEM state
    Eigen::VectorXf m_mean;
    Eigen::VectorXf m_std;
    Eigen::VectorXf m_bestSolution;
    double m_bestReward = -1e9;
    
    // Policy network (for POLICY mode)
    std::unique_ptr<PolicyNetwork> m_policy;
    
    // Random generator
    std::mt19937 m_rng;
    
    std::atomic<bool> m_cancelRequested{false};
};

/**
 * @brief Evolution Strategies (ES) optimizer
 * 
 * OpenAI-style ES with parallel evaluation
 * Better suited for neural network optimization than CEM
 */
struct ESConfig {
    int population_size = 50;
    float learning_rate = 0.01f;
    float noise_std = 0.1f;
    int max_generations = 100;
    int episode_steps = 50;
    float action_scale = 0.1f;
};

class ESOptimizer {
public:
    using Config = ESConfig;
    
    explicit ESOptimizer(BiteSimulator* simulator, const Config& config = Config{});
    
    // Run optimization
    double optimize(std::function<void(int, double, double)> progressCallback = nullptr);
    
    // Get policy
    PolicyNetwork* policy() { return m_policy.get(); }
    double getBestReward() const { return m_bestReward; }
    
    // Cancellation
    void cancel() { m_cancelRequested.store(true); }
    bool isCancelled() const { return m_cancelRequested.load(); }
    void resetCancellation() { m_cancelRequested.store(false); }

private:
    double evaluatePolicy(const PolicyNetwork& policy, int steps);
    RLState getCurrentState() const;
    
    BiteSimulator* m_simulator;
    Config m_config;
    
    std::unique_ptr<PolicyNetwork> m_policy;
    double m_bestReward = -1e9;
    
    std::mt19937 m_rng;
    std::atomic<bool> m_cancelRequested{false};
};

struct Transition {
    Eigen::VectorXf state;
    Eigen::VectorXf action;
    float reward;
    Eigen::VectorXf next_state;
    bool done;
    float log_prob;
    float value;
};

class ActorCriticNetwork {
public:
    ActorCriticNetwork(int state_dim = RLState::dimension(),
                       int action_dim = RLAction::dimension(),
                       int hidden = 64);
    
    Eigen::VectorXf actorForward(const Eigen::VectorXf& state) const;
    float criticForward(const Eigen::VectorXf& state) const;
    
    Eigen::VectorXf getActorParams() const;
    Eigen::VectorXf getCriticParams() const;
    void setActorParams(const Eigen::VectorXf& params);
    void setCriticParams(const Eigen::VectorXf& params);
    int actorParamCount() const;
    int criticParamCount() const;
    
    void randomize(float scale = 0.1f);

private:
    Eigen::MatrixXf m_actorW1, m_actorW2;
    Eigen::VectorXf m_actorB1, m_actorB2;
    Eigen::MatrixXf m_criticW1, m_criticW2;
    Eigen::VectorXf m_criticB1, m_criticB2;
    Eigen::VectorXf m_logStd;
    
    static Eigen::VectorXf tanh_act(const Eigen::VectorXf& x);
};

struct PPOConfig {
    int max_iterations = 50;
    int steps_per_iteration = 64;
    int ppo_epochs = 4;
    int minibatch_size = 32;
    float gamma = 0.99f;
    float gae_lambda = 0.95f;
    float clip_epsilon = 0.2f;
    float actor_lr = 3e-4f;
    float critic_lr = 1e-3f;
    float action_scale = 0.1f;
    float entropy_coef = 0.01f;
};

class PPOOptimizer {
public:
    using Config = PPOConfig;
    
    explicit PPOOptimizer(BiteSimulator* simulator, const Config& config = Config{});
    
    double optimize(std::function<void(int, double, double)> progressCallback = nullptr);
    
    ActorCriticNetwork* network() { return m_network.get(); }
    double getBestReward() const { return m_bestReward; }
    
    void cancel() { m_cancelRequested.store(true); }
    bool isCancelled() const { return m_cancelRequested.load(); }
    void resetCancellation() { m_cancelRequested.store(false); }

private:
    void collectTrajectory(std::vector<Transition>& buffer);
    void computeGAE(std::vector<Transition>& buffer, 
                    std::vector<float>& advantages,
                    std::vector<float>& returns);
    void updatePolicy(const std::vector<Transition>& buffer,
                      const std::vector<float>& advantages,
                      const std::vector<float>& returns);
    
    Eigen::VectorXf sampleAction(const Eigen::VectorXf& mean, float& logProb);
    float computeLogProb(const Eigen::VectorXf& action, const Eigen::VectorXf& mean);
    RLState getCurrentState() const;
    
    BiteSimulator* m_simulator;
    Config m_config;
    std::unique_ptr<ActorCriticNetwork> m_network;
    double m_bestReward = -1e9;
    std::mt19937 m_rng;
    std::atomic<bool> m_cancelRequested{false};
};

#endif // RLOPTIMIZER_H
