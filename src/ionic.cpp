#include <math.h>
#include "ionic.h"
#include <iostream>

using namespace arma;

void ionic::init(potential &HH, double Mass, double vv, double xx, int state, double xl, double xr)
{
	nhops = 0;
	istate = state;
	mass = Mass;
	v_pre = v_new = vv;
	ek = 0.5*mass*vv*vv;
	for(int t1 = 1; t1<HH.nx; t1++)
		if (HH.x(t1) >= xx)
		{
			ind_pre = ind_new = t1;
			break;
		}
	for(int t1 = 1; t1<HH.nx; t1++)
		if (HH.x(t1) >= xl)
		{
			ind_l = t1;
			break;
		}
	ind_r = HH.nx-2;
	for(int t1 = 1; t1<HH.nx; t1++)
		if (HH.x(t1) >= xr)
		{
			ind_r = t1;
			break;
		}
}

void ionic::move(potential& HH)
{
	double aa = HH.F(istate,ind_new)/mass;
	double dtl = -1,dtr = -1, deltal,deltar,t1,t2;
	double tmp;
	//
	v_pre = v_new;
	ind_pre = ind_new;
	//
	deltal = v_pre*v_pre-2*aa*HH.dx;
	deltar = v_pre*v_pre+2*aa*HH.dx;
	// allow right
	if (ek + HH.H_fock(istate,ind_pre) >= HH.H_fock(istate,ind_pre+1) && deltar > 0)
	{
		deltar = sqrt(deltar);
		t1 = (-v_pre + deltar) / aa;
		t2 = (-v_pre - deltar) / aa;
		if (t1 * t2 > 0)
			dtr = fmin(t1,t2);
		else
			dtr = fmax(t1,t2);
	}
	// allow left
	if (ek + HH.H_fock(istate,ind_pre) >= HH.H_fock(istate,ind_pre-1) && deltal > 0)
	{
		deltal = sqrt(deltal);
		t1 = (-v_pre + deltal) / aa;
		t2 = (-v_pre - deltal) / aa;
		if (t1 * t2 > 0)
			dtl = fmin(t1,t2);
		else
			dtl = fmax(t1,t2);
	}
	// choose dt
	if (dtl<0 && dtr<0)
		// have to stay middle
		dt = -2*v_pre/aa;
	else if (dtl*dtr > 0)
		dt = fmin(dtl,dtr);
	else
		dt = fmax(dtl,dtr);
	//
	// set new status
	tmp = 0.5*aa*dt*dt + v_pre*dt;
	if (tmp > 0.5*HH.dx)
	{
		// move right
		ind_new = ind_pre+1;
		ek = HH.H_fock(istate,ind_pre) + ek - HH.H_fock(istate,ind_new);
		v_new = sqrt(2*ek / mass);
	}
	else if (tmp < -0.5*HH.dx)
	{
		// move left
		ind_new = ind_pre-1;
		ek = HH.H_fock(istate,ind_pre) + ek - HH.H_fock(istate,ind_new);
		v_new =-sqrt(2*ek / mass);
	}
	else
	{
		// stay middle
		ind_new = ind_pre;
		v_new = - v_pre;
	}
}

int ionic::check_stop()
{
	if (ind_new <= ind_l && v_new <0)
		return -1;
	if (ind_new >= ind_r && v_new >0)
		return 1;
	return 0;
}

void ionic::try_hop(potential& HH, cx_mat& rho)
{
	// Gij should be 2Re( dt * (dd*v)_ij * rho_ji ) / rho_ii
	cx_mat T = HH.dd.slice(ind_new) * dt * cx_double(v_new,0.0);
	vec rate(HH.dim);
	//
	for (int t1=0; t1<HH.dim; t1++)
	{
		if (t1==istate)
			rate(t1) = 0;
		else
			rate(t1) = real( T(istate,t1)*rho(t1,istate) ) * 2 / real(rho(istate,istate));
		if (rate(t1) < 0)
			rate(t1) = 0;
		if (ek + HH.H_fock(istate,ind_new) < HH.H_fock(t1,ind_new))
			rate(t1) = 0;
	}
	//std::cout<<ind_new<<'\t'<<v_new;
	//for( int t1=0; t1<HH.dim; t1++)
	//	std::cout<<'\t'<<rate(t1);
	//std::cout<<std::endl;
	//
	for (int t1=1; t1<HH.dim; t1++)
		rate(t1) += rate(t1-1);
	//
	vec tmp(1,fill::randu);
	int new_state = istate;
	for (int t1=0; t1<HH.dim; t1++)
		if( tmp(0) < rate(t1) )
		{
			new_state = t1;
			break;
		}
	//
	double ek_new = ek + HH.H_fock(istate,ind_new) - HH.H_fock(new_state,ind_new);
	v_new = v_new * sqrt(ek_new / ek);
	ek = ek_new;
	if (istate != new_state) nhops++;
	istate = new_state;
}

void ionic::try_hop_with_bath_state2(potential& HH, cx_mat& rho)
{
	// look at diagonal of d\rho/dt
	//
	// the system part is [ 2Im( H_ij * rho_ji ) + 2Re( (dd*v)_ij * rho_ji ) ] * dt / rho_ii
	//
	// the bath part together is B_bath * rho, since B_bath is in diabatic basis,
	// need to change basis first:
	//     U\dagger [ B * (U \rho U\dagger) ] U
	//
	// for dim > 2, need to directly change basis for B, wich in tensor form is B_ij^kl, i.e.
	// B(new basis)_ij^kl = U\dagger_ia U\dagger_jb U_kc U_ld B_ab^cd
	// then the term with rho_ij should mean rate from i to j.
	// Q(1): how to deal with rho_ii term
	// Q(2): could there be rho_ab term where neither a nor b is i?
	// 
	if( HH.dim != 2)
	{
		cout<<"The dimension of H must be 2"<<endl;
		exit(EXIT_FAILURE);
	}
	cx_mat T = HH.dd_with_bath.slice(ind_new) * v_new;
	vec rate_s(HH.dim), rate_b(HH.dim), rate(HH.dim*2);
	//
	// system part
	for (int t1=0; t1<HH.dim; t1++)
	{
		if (t1==istate)
			rate_s(t1) = 0;
		else
		{
			rate_s(t1) = real( T(istate,t1)*rho(t1,istate) ) + imag( HH.H_with_bath(istate,t1,ind_new)*rho(t1,istate) );
			rate_s(t1) *= (2*dt)/real(rho(istate,istate));
		}
	}
	//
	// bath part
	cx_mat U = HH.eigvec_with_bath.slice(ind_new);
	cx_mat tmp_rate = U.t()*reshape(HH.B_bath.slice(ind_new)*reshape(U*rho*U.t(),4,1),2,2)*U;
	rate_b(istate) = 0;
	rate_b(1-istate) = -real(tmp_rate(istate,istate))*dt/real(rho(istate,istate));
	//
	// adjust rate, < 0 and energy forbidden
	for(int t1=0; t1<HH.dim; t1++)
	{
		if(ek + HH.eigval_with_bath(istate,ind_new) < HH.eigval_with_bath(t1,ind_new))
		{
			rate(t1) = 0;
			if(rate_s(t1)+rate_b(t1)<0)
				rate(t1+HH.dim) = 0;
			else if(rate_s(t1)<0)
				rate(t1+HH.dim) = rate_s(t1)+rate_b(t1);
			else if(rate_b(t1)<0)
				rate(t1+HH.dim) = 0;
			else
				rate(t1+HH.dim) = rate_b(t1);
		}
		else
		{
			if(rate_s(t1)+rate_b(t1)<0)
				rate(t1) = rate(t1+HH.dim) = 0;
			else if(rate_s(t1)<0)
			{
				rate(t1) = 0;
				rate(t1+HH.dim) = rate_s(t1)+rate_b(t1);
			}
			else if(rate_b(t1)<0)
			{
				rate(t1) = rate_s(t1)+rate_b(t1);
				rate(t1+HH.dim) = 0;
			}
			else
			{
				rate(t1) = rate_s(t1);
				rate(t1+HH.dim) = rate_b(t1);
			}
		}
	}
	for (int t1=1; t1<HH.dim*2; t1++)
		rate(t1) += rate(t1-1);
	// TODO: this should never happen, it means dt is too big, should through out an warning
	if(rate(HH.dim*2-1) > 1)
		rate = rate / rate(HH.dim*2-1);
	//
	vec tmp(1,fill::randu);
	int new_state = istate;
	int rescale_energy=0;
	for (int t1=0; t1<HH.dim*2; t1++)
		if( tmp(0) < rate(t1) )
		{
			new_state = t1%HH.dim;
			if( t1 < HH.dim)
				rescale_energy = 1;
			break;
		}
	//
	double ek_new = ek;
	if (rescale_energy)
	{
		ek_new = ek_new + HH.eigval_with_bath(istate,ind_new) - HH.eigval_with_bath(new_state,ind_new);
		v_new = v_new * sqrt(ek_new / ek);
		ek = ek_new;
	}
	if (istate != new_state) nhops++;
	istate = new_state;
}

void ionic::print_rate(arma::vec& xx, potential& HH, arma::cx_mat& rho)
{
	int sz=HH.dim;
	cx_mat T = HH.dd.slice(ind_new) * dt * cx_double(v_new,0.0);
	cout<<xx(ind_new);
	for (int t1=0; t1<sz-1; t1++)
		for (int t2=t1+1;t2<sz; t2++)
			cout<<'\t'<<2*real(T(t1,t2)*rho(t2,t1));
	cout<<endl;
}
