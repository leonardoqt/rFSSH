#include <math.h>
#include <iomanip>
#include "potential.h"

using namespace arma;

void potential::generate_H(vec X, double E1, double E2, double vdd, double gamma1, double gamma2)
{
	x = X;
	dx = x(1) - x(0);
	nx = x.n_rows;
	Evac = zeros<vec>(nx);
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
	//Hs.slice(nx-1).print("Hs");
	//Ht.slice(nx-1).print("Ht");
}

void potential::generate_H(arma::vec X, double omega, double g0, double Ed, double gamma)
{
	x = X;
	dx = x(1) - x(0);
	nx = x.n_rows;
	Evac = zeros<vec>(nx);
	Hs = cube(sz_s,sz_s,nx,fill::zeros);
	Ht = cube(sz_t,sz_t,nx,fill::zeros);
	Eb = linspace(-dep_bath,dep_bath,nbath);
	double vsb = sqrt( gamma/(datum::pi*nbath/dep_bath) );
	//
	//=========================================
	for (int t1=0; t1<nx; t1++)
	{
		Evac(t1) = 0.5*omega*x(t1)*x(t1);
		Hs(0,0,t1) = 0.5*omega*x(t1)*x(t1) + sqrt(2)*g0*x(t1) + Ed;
		Ht.slice(t1).diag() = join_vert(Hs.slice(t1).diag(),Eb);
		for (int t2=0; t2<nbath; t2++)
		{
			Ht(0,t2+sz_s,t1) = vsb;
			Ht(t2+sz_s,0,t1) = vsb;
		}
	}
	//Hs.slice(nx-1).print("Hs");
	//Ht.slice(nx-1).print("Ht");
}

void potential::diag_H()
{
	switch(sz_s)
	{
	case(1):
		{
			diag_H_1imp();
			break;
		}
	case(2):
		{
			diag_H_2imp();
			break;
		}
	}
}

void potential::diag_H_2imp()
{
	vec tmp;
	eigvec_t = cube(sz_t,sz_t,nx,fill::zeros);
	eigvec_s = cube(sz_s,sz_s,nx,fill::zeros);
	eigval_t = mat(sz_t,nx,fill::zeros);
	eigval_s = mat(sz_s,nx,fill::zeros);
	E_f      = mat(sz_f,nx,fill::zeros);
	F_f      = mat(sz_f,nx,fill::zeros);
	dd       = cube(sz_f,sz_f,nx,fill::zeros);
	//
	for(int t1=0; t1<nx; t1++)
	{
		eig_sym(tmp,eigvec_t.slice(t1),Ht.slice(t1));
		eigval_t.col(t1) = tmp;
		eig_sym(tmp,eigvec_s.slice(t1),Hs.slice(t1));
		eigval_s.col(t1) = tmp;
	}
	// sign correction for eigvec_s
	for(int t1=0; t1<sz_s; t1++)
	{
		for(int t2=1; t2<nx; t2++)
		{
			if (dot(eigvec_s.slice(t2).col(t1),eigvec_s.slice(t2-1).col(t1)) < 0)
				eigvec_s.slice(t2).col(t1) *= -1;
		}
	}
	// E_f
	for(int t1=0; t1<nx; t1++)
	{
		E_f(0,t1) = Evac(t1);
		E_f(1,t1) = eigval_s(0,t1);
		E_f(2,t1) = eigval_s(1,t1);
		E_f(3,t1) = E_f(1,t1) + E_f(2,t1);
	}
	// force
	for(int t1=1; t1<nx-1; t1++)
		F_f.col(t1) = ( E_f.col(t1+1) - E_f.col(t1-1) ) / (-2*dx);
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

void potential::diag_H_1imp()
{
	vec tmp;
	eigvec_t = cube(sz_t,sz_t,nx,fill::zeros);
	eigvec_s = cube(sz_s,sz_s,nx,fill::zeros);
	eigval_t = mat(sz_t,nx,fill::zeros);
	eigval_s = mat(sz_s,nx,fill::zeros);
	E_f      = mat(sz_f,nx,fill::zeros);
	F_f      = mat(sz_f,nx,fill::zeros);
	dd       = cube(sz_f,sz_f,nx,fill::zeros);
	//
	for(int t1=0; t1<nx; t1++)
	{
		eig_sym(tmp,eigvec_t.slice(t1),Ht.slice(t1));
		eigval_t.col(t1) = tmp;
		eig_sym(tmp,eigvec_s.slice(t1),Hs.slice(t1));
		eigval_s.col(t1) = tmp;
	}
	// sign correction for eigvec_s
	for(int t1=0; t1<sz_s; t1++)
	{
		for(int t2=1; t2<nx; t2++)
		{
			if (dot(eigvec_s.slice(t2).col(t1),eigvec_s.slice(t2-1).col(t1)) < 0)
				eigvec_s.slice(t2).col(t1) *= -1;
		}
	}
	// E_f
	for(int t1=0; t1<nx; t1++)
	{
		E_f(0,t1) = Evac(t1);
		E_f(1,t1) = eigval_s(0,t1);
	}
	// force
	for(int t1=1; t1<nx-1; t1++)
		F_f.col(t1) = ( E_f.col(t1+1) - E_f.col(t1-1) ) / (-2*dx);
	// derivative coupling
	// it is all zero, nothing to calculate
}
