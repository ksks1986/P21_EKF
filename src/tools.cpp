#include <iostream>
#include <cfloat>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

	VectorXd rmse(4);
	rmse << 0,0,0,0;

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	//  * the estimation vector size should equal ground truth vector size
	// ... your code here
	if(estimations.size() == 0){
	    cout << "Estimation vector size is zero." << endl;
	    return rmse;
	}
	if(estimations.size() != ground_truth.size()){
	    cout << "Estimtion vector size isn't match ground truth vector size." << endl;
	    return rmse;
	}
	

	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
    		VectorXd tmp(4);
        	tmp =  (ground_truth[i] - estimations[i]);
		tmp = tmp.array() * tmp.array();
       		rmse += tmp;
	}

	//calculate the mean
    	rmse /= estimations.size();

	//calculate the squared root
    	rmse = rmse.array().sqrt();

	return rmse;
}



MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	float absp  = sqrt(px*px + py*py);
    	float absp2 = absp * absp;
    	float absp3 = absp2 * absp; 

	//check division by zero
	if (absp3 < FLT_EPSILON){
	    cout << "Error: Division by zero." << endl;
	    Hj << 0, 0, 0, 0,
		  0, 0, 0, 0,
		  0, 0, 0, 0;
	    return Hj;
	}
	
	//compute the Jacobian matrix
	float H11 =  px / absp;
    	float H12 =  py / absp;
    	float H21 = -py / absp2;
    	float H22 =  px / absp2;
    	float H31 =  py * (vx*py - vy*px) / absp3;
    	float H32 =  px * (vy*px - vx*py) / absp3;
    	float H33 =  H11;
    	float H34 =  H12;
    
    	Hj << H11, H12, 0.0, 0.0,
	      H21, H22, 0.0, 0.0,
	      H31, H32, H33, H34;
          
	return Hj;
}

