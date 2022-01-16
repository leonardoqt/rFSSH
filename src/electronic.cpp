#include "electronic.h"
#include <iostream>

using namespace arma;

void electronic::init_rho(mat rho0_s, potential& HH, double beta)
{
	vec N_b = 1/(1+exp(beta * HH.Eb));
	N_t = zeros<cx_mat>(HH.sz_t,HH.sz_t);
	N_s = zeros<cx_mat>(HH.sz_s,HH.sz_s);
	rho_fock = zeros<cx_mat>(HH.sz_fock,HH.sz_fock);
	N_t.diag() = cx_vec (join_vert(zeros<vec>(HH.sz_s),N_b),zeros<vec>(HH.sz_t) );
	N_t(span(0,1),span(0,1)) = cx_mat (rho0_s,zeros<mat>(HH.sz_s,HH.sz_s));
	//
	N_s = N_t(span(0,1),span(0,1));
	//
	rho_fock(3,3) = N_s(0,0) * N_s(1,1);
	rho_fock(1,1) = N_s(0,0) - rho_fock(3,3);
	rho_fock(2,2) = N_s(1,1) - rho_fock(3,3);
	rho_fock(0,0) = cx_double(1,0) - N_s(0,0) - N_s(1,1) + rho_fock(3,3);
	rho_fock(1,2) = N_s(0,1);
	rho_fock(2,1) = N_s(1,0);
}

void electronic::evolve(potential& HH, ionic& AA)
{
	mat U_t = HH.eigvec_t.slice(AA.ind_new);
	mat U_s = HH.eigvec_s.slice(AA.ind_new);
	cx_mat d0_da = U_t.t() * eye(HH.sz_t,HH.sz_s);
	cx_mat N0_a = U_t.t() * N_t * U_t;
	cx_mat dt_da = d0_da;
	cx_mat phase = exp(cx_double(1,0)*HH.eigval_t.col(AA.ind_new)*AA.dt);
	for (int t1=0; t1<HH.sz_t; t1++)
		dt_da.row(t1) = phase(t1) * d0_da.row(t1);
	cx_mat dt_aa = dt_da * U_s;
	//
	// N_s
	N_s = dt_aa.t() * N0_a * dt_aa;
	//
	// N_t
	cx_mat U_t_phase = zeros<cx_mat>(HH.sz_t,HH.sz_t);
	for (int t1=0; t1<HH.sz_t; t1++)
		U_t_phase.col(t1) = phase(t1) * U_t.col(t1);
	cx_mat EHT = U_t_phase * U_t.t();
	N_t = EHT.t() * N_t * EHT;
	//
	// rho_fock
	rho_fock(3,3) = N_s(0,0) * N_s(1,1);
	rho_fock(1,1) = N_s(0,0) - rho_fock(3,3);
	rho_fock(2,2) = N_s(1,1) - rho_fock(3,3);
	rho_fock(0,0) = cx_double(1,0) - N_s(0,0) - N_s(1,1) + rho_fock(3,3);
	rho_fock(1,2) = N_s(0,1);
	rho_fock(2,1) = N_s(1,0);
}

//void electronic::try_decoherence(ionic& AA)
//{
//	if (AA.v_pre*AA.v_new <0 && AA.istate >0)
//	{
//		psi = psi * 0;
//		psi(AA.istate) = cx_double(1,0);
//		rho = psi * psi.t();
//	}
//}
