#include "AdamOptimizer.h"
#include <iostream>

AdamOptimizer::AdamOptimizer(float beta1, float beta2, float epsilon)
    : m_beta1(beta1)
    , m_beta2(beta2)
    , m_epsilon(epsilon)
    , m_m(Eigen::Matrix<float, 6, 1>::Zero())
    , m_v(Eigen::Matrix<float, 6, 1>::Zero())
    , m_t(0)
{
}

Eigen::Matrix<float, 6, 1> AdamOptimizer::step(const Eigen::Matrix<float, 6, 1>& gradient, float learningRate)
{
    m_t++;

    // Update biased first moment estimate (momentum)
    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    m_m = m_beta1 * m_m + (1.0f - m_beta1) * gradient;

    // Update biased second raw moment estimate (RMSprop)
    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_v = m_beta2 * m_v + (1.0f - m_beta2) * gradient.array().square().matrix();

    // Compute bias-corrected first moment estimate
    // m_hat_t = m_t / (1 - beta1^t)
    Eigen::Matrix<float, 6, 1> m_hat = m_m / (1.0f - std::pow(m_beta1, m_t));

    // Compute bias-corrected second raw moment estimate
    // v_hat_t = v_t / (1 - beta2^t)
    Eigen::Matrix<float, 6, 1> v_hat = m_v / (1.0f - std::pow(m_beta2, m_t));

    // Compute parameter update
    // theta_t = theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)
    Eigen::Matrix<float, 6, 1> update = learningRate * m_hat.array() /
                                     (v_hat.array().sqrt() + m_epsilon);

    return update;
}

void AdamOptimizer::reset()
{
    m_m = Eigen::Matrix<float, 6, 1>::Zero();
    m_v = Eigen::Matrix<float, 6, 1>::Zero();
    m_t = 0;
}
