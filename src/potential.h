#ifndef __POTENTIAL__
#define __POTENTIAL__

#include <armadillo>

class potential; // time independent part

class potential
{
private:
	const int nbath = 100;
	const double dep_bath = 0.3;
	void diag_H_1imp();
	void diag_H_2imp();
public:
	const int sz_s = 1;
	const int sz_f = 1<<sz_s; // this is 2^sz_s
	const int sz_t = sz_s + nbath;
	double dx;
	int nx;
	arma::vec x;
	arma::vec Eb;
	arma::vec Evac;
	arma::cube Ht, Hs;
	arma::cube eigvec_t, eigvec_s;
	arma::mat eigval_t, eigval_s, F_f, E_f;
	arma::cube dd;
	void generate_H(arma::vec X, double E1, double E2, double vdd, double gamma1, double gamma2);
	void generate_H(arma::vec X, double omega, double g0, double Ed, double gamma);
	//
	void diag_H();
};

#endif
