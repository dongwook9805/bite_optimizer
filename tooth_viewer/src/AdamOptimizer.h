#ifndef ADAMOPTIMIZER_H
#define ADAMOPTIMIZER_H

#include <Eigen/Core>
#include <cmath>

/**
 * @brief Adam Optimizer for 6-DOF rigid body optimization
 *
 * Adam combines:
 * - Momentum (1st moment, beta1): remembers gradient direction
 * - RMSprop (2nd moment, beta2): adapts learning rate per parameter
 * - Bias correction: accounts for initialization bias
 *
 * References: Kingma & Ba (2014) "Adam: A Method for Stochastic Optimization"
 */
class AdamOptimizer
{
public:
    /**
     * @brief Construct Adam optimizer
     * @param beta1 Exponential decay rate for first moment (default: 0.9)
     * @param beta2 Exponential decay rate for second moment (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     */
    explicit AdamOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f);

    /**
     * @brief Perform one Adam step
     * @param gradient 6-element gradient vector [rot_x, rot_y, rot_z, trans_x, trans_y, trans_z]
     * @param learningRate Step size (default: 0.01)
     * @return 6-element delta to apply [drot_x, drot_y, drot_z, dtrans_x, dtrans_y, dtrans_z]
     */
    Eigen::Matrix<float, 6, 1> step(const Eigen::Matrix<float, 6, 1>& gradient, float learningRate = 0.01f);

    /**
     * @brief Reset optimizer state (clear momentum moments)
     */
    void reset();

    /**
     * @brief Get current timestep count
     */
    int timestep() const { return m_t; }

private:
    // Adam parameters
    float m_beta1;   // First moment decay (momentum)
    float m_beta2;   // Second moment decay (RMSprop)
    float m_epsilon;  // Numerical stability

    // Internal state
    Eigen::Matrix<float, 6, 1> m_m;  // First moment (moving average of gradients)
    Eigen::Matrix<float, 6, 1> m_v;  // Second moment (moving average of squared gradients)
    int m_t;              // Timestep counter
};

#endif // ADAMOPTIMIZER_H
