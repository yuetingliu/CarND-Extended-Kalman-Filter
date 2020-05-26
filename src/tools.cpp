#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  // check inputs
  // estimations and ground_truth should have the same size and not zero length
  if (estimations.size() != ground_truth.size()) {
    cout << "Estimations and ground truth data should have the same length" << endl;
    return rmse;
  }
  if (estimations.size() == 0) {
    cout << "No estimation data!!!" << endl;
    return rmse;
  }

  // iterate through all estimation data
  for (unsigned int i=0; i < estimations.size(); i++) {
    // calculate residue
    VectorXd res = estimations[i] - ground_truth[i];
    // square residue and add to rmse
    res = res.array() * res.array();
    rmse += res;
  }

  // calculate mean
  rmse /= estimations.size();

  // get square root
  rmse = rmse.array().sqrt();
  cout << "rmse: " << rmse << endl;

  return rmse;
}



MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3, 4);
  Hj << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

  // First get state parameters: px, py, vx, vy
  float px = x_state[0];
  float py = x_state[1];
  float vx = x_state[2];
  float vy = x_state[3];

  // precompute some terms to prevent repeated computations and for readability
  float c1 = px*px + py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  // check possible zero division
  if (fabs(c1) < 0.0001) {
    cout << "Error, division by zero" << endl;
    return Hj;
  }

  // compute Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px)/c1, 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;

}
