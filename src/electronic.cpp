#include "electronic.h"
#include <iostream>

using namespace arma;

// TODO: may change from init rho0_s to rho0_fock
void electronic::init_rho(mat rho0_s, potential& HH, double Beta)
{
	beta = Beta;
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
	//
	rho_fock_old = rho_fock;
	//
	hop_bath = zeros<mat>(HH.sz_fock,HH.sz_fock);
	//real(N_s).print("N_s");
	//real(N_t).print("N_t");
	//real(rho_fock).print("rho_fock");
}

void electronic::evolve(potential& HH, ionic& AA)
{
	cx_double ii = cx_double(0,1);
	mat U_t = HH.eigvec_t.slice(AA.ind_pre);
	mat U_s = HH.eigvec_s.slice(AA.ind_pre);
	mat U_s2= HH.eigvec_s.slice(AA.ind_new);
	// TODO: make sure we want the new adiabats (after dt not before dt, which is U_s)
	cx_mat d0_ad = cx_mat(eye(HH.sz_t,HH.sz_s),zeros<mat>(HH.sz_t,HH.sz_s)) * U_s2;
	cx_vec phase = exp(ii*HH.eigval_t.col(AA.ind_pre)*AA.dt);
	//
	// N_t
	cx_mat U_t_phase = zeros<cx_mat>(HH.sz_t,HH.sz_t);
	for (int t1=0; t1<HH.sz_t; t1++)
		U_t_phase.col(t1) = phase(t1) * U_t.col(t1);
	cx_mat EHT = U_t_phase * U_t.t();
	N_t = EHT.t() * N_t * EHT;
	//
	// N_s
	N_s = d0_ad.t() * N_t * d0_ad;
	//
	// rho_fock and drho
	rho_fock_old = rho_fock;
	rho_fock(3,3) = N_s(0,0) * N_s(1,1);
	rho_fock(1,1) = N_s(0,0) - rho_fock(3,3);
	rho_fock(2,2) = N_s(1,1) - rho_fock(3,3);
	rho_fock(0,0) = cx_double(1,0) - N_s(0,0) - N_s(1,1) + rho_fock(3,3);
	rho_fock(1,2) = N_s(0,1);
	rho_fock(2,1) = N_s(1,0);
	//
	//rho_fock.print("rho_fock");
	//rho_fock_old.print("rho_fock_old");
	//exit(EXIT_FAILURE);
	drho = ( rho_fock - rho_fock_old ) / AA.dt;
	//
	// drho_2fit, it equals drho/dt + i[H, rho] + [T, rho]
	// the evolution use H old, but basis transform should be avarged
	mat hh = diagmat(HH.H_fock.col(AA.ind_pre));
	mat dd = (HH.dd.slice(AA.ind_pre) + HH.dd.slice(AA.ind_new))/2;
	drho_2fit  = drho;
	drho_2fit +=       ii * ( hh*rho_fock_old - rho_fock_old*hh );
	drho_2fit += AA.v_pre * ( dd*rho_fock_old - rho_fock_old*dd );
	//drho.print("drho");
	//drho_2fit.print("drho2fit");
	//exit(EXIT_FAILURE);
}

void electronic::fit_drho_v1(potential& HH, ionic& AA)
{
	//
	//   0,  1,  2,  3,  4,  5,  6,  7
	// L01,L10,L23,L32,L02,L20,L13,L31;
	// L01 -> |0><1|, which hops from 1 to 0
	cube LL(4,4,8,fill::zeros);
	mat L;
	LL(0,1,0) = LL(1,0,1) = LL(2,3,2) = LL(3,2,3) = 1;
	LL(0,2,4) = LL(2,0,5) = LL(1,3,6) = LL(3,1,7) = 1;
	//
	cx_mat rho_dot1(4,4,fill::zeros), rho_dot2(4,4,fill::zeros);
	// lambda_01/lambda_10 = exp(beta*E), i.e.,
	// lambda_10 = lambda_01 * exp(-beta*E)
	L = LL.slice(0);
	rho_dot1 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(1);
	rho_dot1 += exp(-beta*HH.eigval_s(0,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	L = LL.slice(2);
	rho_dot1 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(3);
	rho_dot1 += exp(-beta*HH.eigval_s(0,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	//
	L = LL.slice(4);
	rho_dot2 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(5);
	rho_dot2 += exp(-beta*HH.eigval_s(1,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	L = LL.slice(6);
	rho_dot2 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(7);
	rho_dot2 += exp(-beta*HH.eigval_s(1,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	//
	// fitting: write rho as a column vector, V = drho_2fit, M = [rho_dot1, rho_dot2]
	// then Lambda = (M\dagger M)^(-1)*Re(M\dagger V), where Lambda = [l1; l2]
	cx_vec V = drho_2fit.as_col();
	cx_mat M = join_horiz(rho_dot1.as_col(),rho_dot2.as_col());
	mat MM = real(M.t()*M);
	vec MV = real(M.t()*V);
	vec Lambda = MM.i()*MV;
	//reshape(V,4,4).print("target");
	//reshape(M*Lambda-V,4,4).print("res");
	//exit(EXIT_FAILURE);
	//
	// put results into hop_bath, where hop_bath(i,j) is from i to j, i.e., |j><i|
	hop_bath = zeros<mat>(4,4);
	hop_bath(0,1) = Lambda(0)*exp(-beta*HH.eigval_s(0,AA.ind_pre)); hop_bath(1,0) = Lambda(0);
	hop_bath(2,3) = Lambda(0)*exp(-beta*HH.eigval_s(0,AA.ind_pre)); hop_bath(3,2) = Lambda(0);
	hop_bath(0,2) = Lambda(1)*exp(-beta*HH.eigval_s(1,AA.ind_pre)); hop_bath(2,0) = Lambda(1);
	hop_bath(1,3) = Lambda(1)*exp(-beta*HH.eigval_s(1,AA.ind_pre)); hop_bath(3,1) = Lambda(1);
}

void electronic::fit_drho_v2(potential& HH, ionic& AA)
{
	//
	//   0,  1,  2,  3,  4,  5,  6,  7
	// L01,L10,L23,L32,L02,L20,L13,L31;
	// L01 -> |0><1|, which hops from 1 to 0
	cube LL(4,4,8,fill::zeros);
	mat L;
	LL(0,1,0) = LL(1,0,1) = LL(2,3,2) = LL(3,2,3) = 1;
	LL(0,2,4) = LL(2,0,5) = LL(1,3,6) = LL(3,1,7) = 1;
	//
	cx_mat rho_dot0(4,4,fill::zeros);
	cx_mat rho_dot1(4,4,fill::zeros);
	cx_mat rho_dot2(4,4,fill::zeros);
	cx_mat rho_dot3(4,4,fill::zeros);
	// lambda_01/lambda_10 = exp(beta*E), i.e.,
	// lambda_10 = lambda_01 * exp(-beta*E)
	L = LL.slice(0);
	rho_dot0 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(1);
	rho_dot0 += exp(-beta*HH.eigval_s(0,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	L = LL.slice(2);
	rho_dot1 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(3);
	rho_dot1 += exp(-beta*HH.eigval_s(0,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	//
	L = LL.slice(4);
	rho_dot2 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(5);
	rho_dot2 += exp(-beta*HH.eigval_s(1,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	L = LL.slice(6);
	rho_dot3 += L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2;
	L = LL.slice(7);
	rho_dot3 += exp(-beta*HH.eigval_s(1,AA.ind_pre)) * ( L*rho_fock_old*L.t() - (L.t()*L*rho_fock_old + rho_fock_old*L.t()*L)/2 );
	//
	// fitting: write rho as a column vector, V = drho_2fit, M = [rho_dot1, rho_dot2]
	// then Lambda = (M\dagger M)^(-1)*Re(M\dagger V), where Lambda = [l1; l2]
	cx_vec V = drho_2fit.as_col();
	cx_mat M = join_horiz(rho_dot0.as_col(),rho_dot1.as_col(),rho_dot2.as_col(),rho_dot3.as_col());
	mat MM = real(M.t()*M);
	vec MV = real(M.t()*V);
	vec Lambda = pinv(MM+1e-8*eye(4,4))*MV;
	//reshape(V,4,4).print("target");
	//reshape(M*Lambda-V,4,4).print("res");
	//exit(EXIT_FAILURE);
	//
	// put results into hop_bath, where hop_bath(i,j) is from i to j, i.e., |j><i|
	hop_bath = zeros<mat>(4,4);
	hop_bath(0,1) = Lambda(0)*exp(-beta*HH.eigval_s(0,AA.ind_pre)); hop_bath(1,0) = Lambda(0);
	hop_bath(2,3) = Lambda(1)*exp(-beta*HH.eigval_s(0,AA.ind_pre)); hop_bath(3,2) = Lambda(1);
	hop_bath(0,2) = Lambda(2)*exp(-beta*HH.eigval_s(1,AA.ind_pre)); hop_bath(2,0) = Lambda(2);
	hop_bath(1,3) = Lambda(3)*exp(-beta*HH.eigval_s(1,AA.ind_pre)); hop_bath(3,1) = Lambda(3);
	//vec eigval;
	//mat tmp;
	//eig_sym(eigval,tmp,MM);
	//cout<<eigval(0)<<'\t'<<Lambda(1)<<'\t'<<Lambda(2)<<'\t'<<Lambda(3)<<'\t'<<eigval(0)<<'\t'<<eigval(1)<<'\t'<<eigval(2)<<'\t'<<eigval(3)<<endl;
	//real(drho_2fit.diag()).t().print();
	//real(rho_fock.diag()).t().print();
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
