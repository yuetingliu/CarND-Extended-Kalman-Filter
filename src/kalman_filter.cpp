#include "kalman_filter.h"
#include <math.h>
#include <iostream>

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

/*
 * Please note that the Eigen library does not initialize
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &RL_in,
                        MatrixXd &RR_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_laser = RL_in;
  R_radar = RR_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y_ = z - H_ * x_;
  MatrixXd H_T = H_.transpose();
  MatrixXd S_ = H_ * P_ * H_T + R_laser;
  MatrixXd K_ = P_ * H_T * S_.inverse();

  // update state
  x_ = x_ + (K_ * y_);
  MatrixXd I_;
  int x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
  P_ = (I_ - K_ * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // convert Cartesian coordinates to polar coordinates
  // get Cartesian coordinates
  float px = x_[0];
  float py = x_[1];
  float vx = x_[2];
  float vy = x_[3];

  // calculate hx
  int size_z = z.size();
  VectorXd hx(size_z);
  // pre-compute some terms to prevent repeated computation
  float c1 = sqrt(px*px + py*py);
  if (fabs(c1) < 0.0001) {
    cout << "Zero division error" << endl;
    return;
  }
  float phi = atan(py/px);
  hx << c1, phi, (px*vx + py*vy)/c1;

  // calcuate y
  VectorXd y_ = z - hx;
  // make sure phi in y_ is in range(-pi, pi)
  if (y_[1] < -M_PI) {
    y_[1]  += 2*M_PI;
  } else if (y_[1] > M_PI){
    y_[1]  -= M_PI;
  }

  // calculate Jacobian matrix
  MatrixXd Hj = tool.CalculateJacobian(x_);

  // calculate S, K
  MatrixXd HjT = Hj.transpose();
  MatrixXd S_ = Hj * P_ * HjT + R_radar;
  MatrixXd K_ = P_ * HjT * S_.inverse();

  // update
  x_ = x_ + (K_ * y_);
  MatrixXd I_;
  int x_size = x_.size();
  I_ = MatrixXd::Identity(x_size, x_size);
  P_ = (I_ - K_ * Hj) * P_;
}
