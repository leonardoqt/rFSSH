#ifndef __POTENTIAL__
#define __POTENTIAL__

#include <armadillo>

class potential; // time independent part

class potential
{
private:
	const int nbath = 1300;
	const double dep_bath = 3.0;
public:
	const int sz_s = 2;
	const int sz_fock = 4; // this is 2^dim
	int sz_t;
	double dx;
	int nx;
	arma::vec x;
	arma::vec Eb;
	arma::cube Ht, Hs;
	arma::cube eigvec_t, eigvec_s;
	arma::mat eigval_t, eigval_s, F, H_fock;
	arma::cube dd;
	void generate_H(arma::vec X, double E1, double E2, double vdd, double gamma1, double gamma2);
	//
	void diag_H();
};

#endif
