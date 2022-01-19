#include <math.h>
#include <iomanip>
#include "potential.h"

using namespace arma;

void potential::generate_H(vec X, double E1, double E2, double vdd, double gamma1, double gamma2)
{
	sz_t = sz_s + nbath;
	x = X;
	dx = x(1) - x(0);
	nx = x.n_rows;
	Hs = cube(sz_s,sz_s,nx,fill::zeros);
	Ht = cube(sz_t,sz_t,nx,fill::zeros);
	Eb = linspace(-dep_bath,dep_bath,nbath);
	double vsb1 = sqrt( gamma1/(datum::pi*nbath/dep_bath) );
	double vsb2 = sqrt( gamma2/(datum::pi*nbath/dep_bath) );
	//
	//=========================================
	for (int t1=0; t1<nx; t1++)
	{
		Hs(0,0,t1) = E1;
		Hs(1,1,t1) = E2;
		Hs(0,1,t1) = Hs(1,0,t1) = vdd;
		Ht.slice(t1).diag() = join_vert(Hs.slice(t1).diag(),Eb);
		Ht(0,1,t1) = Ht(1,0,t1) = vdd;
		for (int t2=0; t2<nbath; t2++)
		{
			Ht(0,t2+sz_s,t1) = vsb1;
			Ht(1,t2+sz_s,t1) = vsb2;
			Ht(t2+sz_s,0,t1) = vsb1;
			Ht(t2+sz_s,1,t1) = vsb2;
		}
	}
}

void potential::diag_H()
{
	vec tmp;
	eigvec_t = cube(sz_t,sz_t,nx,fill::zeros);
	eigvec_s = cube(sz_s,sz_s,nx,fill::zeros);
	eigval_t = mat(sz_t,nx,fill::zeros);
	eigval_s = mat(sz_s,nx,fill::zeros);
	H_fock   = mat(sz_fock,nx,fill::zeros);
	F        = mat(sz_fock,nx,fill::zeros);
	dd       = cube(sz_fock,sz_fock,nx,fill::zeros);
	//
	for(int t1=0; t1<nx; t1++)
	{
		eig_sym(tmp,eigvec_t.slice(t1),Ht.slice(t1));
		eigval_t.col(t1) = tmp;
		eig_sym(tmp,eigvec_s.slice(t1),Hs.slice(t1));
		eigval_s.col(t1) = tmp;
	}
	// sign correction for eigvec
	for(int t1=0; t1<sz_s; t1++)
	{
		for(int t2=1; t2<nx; t2++)
		{
			if (dot(eigvec_s.slice(t2).col(t1),eigvec_s.slice(t2-1).col(t1)) < 0)
				eigvec_s.slice(t2).col(t1) *= -1;
		}
	}
	// H_fock
	for(int t1=0; t1<nx; t1++)
	{
		H_fock(0,t1) = 0;
		H_fock(1,t1) = eigval_s(0,t1);
		H_fock(2,t1) = eigval_s(1,t1);
		H_fock(3,t1) = H_fock(1,t1) + H_fock(2,t1);
	}
	// force
	for(int t1=1; t1<nx-1; t1++)
		F.col(t1) = ( H_fock.col(t1+1) - H_fock.col(t1-1) ) / (-2*dx);
	// derivative coupling
	mat tmp_dd;
	for(int t1=1; t1<nx-1; t1++)
	{
		//TODO: use log to calculate dd
		tmp_dd = (eigvec_s.slice(t1-1).t()*eigvec_s.slice(t1+1) - eye(sz_s,sz_s)) / (2*dx);
		tmp_dd = ( tmp_dd - tmp_dd.t() ) / 2;
		dd(1,2,t1) = tmp_dd(0,1);
		dd(2,1,t1) = tmp_dd(1,0);
	}
}
