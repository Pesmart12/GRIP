#pragma once

#include <functional>

#include <Eigen/Core>

namespace grip::testutil {

// Central-difference Jacobian of f: R^InDim -> R^OutDim at x.
//
// eps trades off truncation error (O(eps^2) for central differences)
// against floating-point cancellation error in (f(x+eps)-f(x-eps))
// (O(machine_eps/eps)). The default balances the two for double
// precision inputs at O(1) scale: eps ~ machine_eps^(1/3) ~ 6e-6.
// See docs/derivations/integrator_jacobians.md.
template <int OutDim, int InDim>
Eigen::Matrix<double, OutDim, InDim> CentralDifferenceJacobian(const std::function<Eigen::Matrix<double, OutDim, 1>(const Eigen::Matrix<double, InDim, 1>&)>& f, const Eigen::Matrix<double, InDim, 1>& x, double eps = 1e-6) {
  Eigen::Matrix<double, OutDim, InDim> jacobian;
  for (int i = 0; i < InDim; ++i) {
    Eigen::Matrix<double, InDim, 1> x_plus = x;
    Eigen::Matrix<double, InDim, 1> x_minus = x;
    x_plus(i) += eps;
    x_minus(i) -= eps;
    jacobian.col(i) = (f(x_plus) - f(x_minus)) / (2.0 * eps);
  }
  return jacobian;
}

// Dynamic-size sibling of the above, for systems where the dimension
// (e.g. 6*N bodies) isn't known at compile time. Same central-difference
// logic and the same eps rationale.
inline Eigen::MatrixXd CentralDifferenceJacobianXd(const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& f, const Eigen::VectorXd& x, double eps = 1e-6) {
  const Eigen::VectorXd f0 = f(x);
  Eigen::MatrixXd jacobian(f0.size(), x.size());
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    Eigen::VectorXd x_plus = x;
    Eigen::VectorXd x_minus = x;
    x_plus(i) += eps;
    x_minus(i) -= eps;
    jacobian.col(i) = (f(x_plus) - f(x_minus)) / (2.0 * eps);
  }
  return jacobian;
}

}  // namespace grip::testutil
