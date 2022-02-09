#ifndef __IONIC__
#define __IONIC__

#include <armadillo>
#include "potential.h"

class ionic;

// In this scheme the ion only presents on x grid, so
// only index of the grid point is recorded.
class ionic
{
public:
	int ind_pre, ind_new; // index on 
	int ind_l, ind_r; // threshold for move left/right
	int istate;
	int nhops;
	double mass, ek, v_pre, v_new;
	double dt;
	//
	void init(potential& HH, double Mass, double vv, double xx, int state, double xl, double xr);
	void move(potential& HH);
	void try_hop(potential& HH, arma::cx_mat& rho, arma::mat& hop_bath);
	void print_rate(arma::vec& xx, potential& HH, arma::cx_mat& rho);
	int check_stop();
};

#endif
