#ifndef __ELECTRONIC_
#define __ELECTRONIC_

#include <armadillo>
#include "potential.h"
#include "ionic.h"

class electronic;

class electronic
{
public:
	int sz_s, sz_t, sz_f;
	arma::cx_mat N_t, N_s, rho_fock, rho_fock_old; // N_t in diabats, N_s in adiabats, rho_fock in adiabats. N_t and N_s are really Tr[rho d_i\dagger d_j]
	arma::cx_mat drho, drho_2fit;
	double beta;
	arma::mat hop_bath;
	//
	void init_rho(arma::mat N0_s, potential& HH, double Beta);
	void evolve(potential& HH, ionic& AA); // update N_t, N_s
	void fit_drho(potential& HH, ionic& AA, int method);
	//void try_decoherence(ionic& AA);
private:
	void construct_rho_fock();
	void fit_drho_v1(potential& HH, ionic& AA); // impose detailed balance, fit only diagonal
	void fit_drho_v2(potential& HH, ionic& AA); // impose detailed balance, fit full matrix
	void fit_drho_v3(); // impose single orbital the same, fit only diagonal
	void fit_drho_v3_1imp(); // impose single orbital the same, fit only diagonal
};
#endif
