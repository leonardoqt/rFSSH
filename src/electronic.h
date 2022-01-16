#ifndef __ELECTRONIC_
#define __ELECTRONIC_

#include <armadillo>
#include "potential.h"
#include "ionic.h"

class electronic;

class electronic
{
public:
	arma::cx_mat N_t, N_s, rho_fock; // N_t in diabats, N_s in adiabats, rho_fock in adiabats. N_t and N_s are really Tr[rho d_i\dagger d_j]
	//
	void init_rho(arma::mat rho0_s, potential& HH, double beta);
	void evolve(potential& HH, ionic& AA); // update N_t, N_s
	void match_drho();
	//void try_decoherence(ionic& AA);
};
#endif
