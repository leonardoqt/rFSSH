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
	etot = ek + HH.E_f(istate,ind_new);
}

void ionic::move(potential& HH)
{
	double aa = HH.F_f(istate,ind_new)/mass;
	double dtl = -1,dtr = -1, deltal,deltar,t1,t2;
	double tmp;
	//
	if ( abs(aa) < 1e-13)
	{
		v_pre = v_new;
		ind_pre = ind_new;
		dt = HH.dx / abs(v_pre);
		if ( v_pre < 0 )
			ind_new--;
		else
			ind_new++;
	}
	else
	{
		v_pre = v_new;
		ind_pre = ind_new;
		//
		deltal = v_pre*v_pre-2*aa*HH.dx;
		deltar = v_pre*v_pre+2*aa*HH.dx;
		// allow right
		if (ek + HH.E_f(istate,ind_pre) >= HH.E_f(istate,ind_pre+1) && deltar > 0)
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
		if (ek + HH.E_f(istate,ind_pre) >= HH.E_f(istate,ind_pre-1) && deltal > 0)
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
			ek = HH.E_f(istate,ind_pre) + ek - HH.E_f(istate,ind_new);
			v_new = sqrt(2*ek / mass);
		}
		else if (tmp < -0.5*HH.dx)
		{
			// move left
			ind_new = ind_pre-1;
			ek = HH.E_f(istate,ind_pre) + ek - HH.E_f(istate,ind_new);
			v_new =-sqrt(2*ek / mass);
		}
		else
		{
			// stay middle
			ind_new = ind_pre;
			v_new = - v_pre;
		}
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

void ionic::try_hop(potential& HH, cx_mat& rho, mat& hop_bath)
{
	// Gij should be 2Re( dt * (dd*v)_ij * rho_ji ) / rho_ii
	cx_mat T = HH.dd.slice(ind_pre) * dt * cx_double(v_pre,0.0);
	vec rate_s(HH.sz_f);
	vec rate_b(HH.sz_f);
	//
	// TODO: rethink about this part
	//*************
	/*
	// rho_ii_dot can be splitted into two terms, the ration decides whether it undergoes the derivative coupling procedue or relaxation procedure
	cx_mat rho_dot1 = (T*rho - rho*T)/dt;
	double rho_ii_dot1 = real(rho_dot1(istate,istate));
	double rho_ii_dot2 = sum(hop_bath.row(istate))*real(rho(istate,istate));
	*/
	//*************
	//
	for (int t1=0; t1<HH.sz_f; t1++)
	{
		if (t1==istate)
		{
			rate_s(t1) = 0;
			rate_b(t1) = 0;
		}
		else
		{
			rate_s(t1) = real( T(istate,t1)*rho(t1,istate) ) * 2 / real(rho(istate,istate));
			rate_b(t1) = ( hop_bath(istate,t1)*real(rho(istate,istate))-hop_bath(t1,istate)*real(rho(t1,t1)) )/real(rho(istate,istate)) * dt;
		}
		if (rate_s(t1) < 0)
			rate_s(t1) = 0;
		if (rate_b(t1) < 0)
			rate_b(t1) = 0;
		if (ek + HH.E_f(istate,ind_new) < HH.E_f(t1,ind_new))
			rate_s(t1) = 0;
	}
	//rate_s.t().print();
	//rate_b.t().print();
	//cout<<endl;
	//
	vec rate = join_vert(rate_s,rate_b);
	for (int t1=1; t1<HH.sz_f*2; t1++)
		rate(t1) += rate(t1-1);
	//
	vec tmp(1,fill::randu);
	int new_state = istate;
	int from_bath = 0;
	for (int t1=0; t1<HH.sz_f*2; t1++)
		if( tmp(0) < rate(t1) )
		{
			new_state = t1 % HH.sz_f;
			from_bath = t1 / HH.sz_f;
			break;
		}
	//
	// adjust velocity
	if (from_bath == 0)
	{
		double ek_new = ek + HH.E_f(istate,ind_new) - HH.E_f(new_state,ind_new);
		v_new = v_new * sqrt(ek_new / ek);
		ek = ek_new;
	}
	if (istate != new_state) nhops++;
	istate = new_state;
	etot = ek + HH.E_f(istate,ind_new);
}

//void ionic::print_rate(arma::vec& xx, potential& HH, arma::cx_mat& rho)
//{
//	int sz=HH.dim;
//	cx_mat T = HH.dd.slice(ind_new) * dt * cx_double(v_new,0.0);
//	cout<<xx(ind_new);
//	for (int t1=0; t1<sz-1; t1++)
//		for (int t2=t1+1;t2<sz; t2++)
//			cout<<'\t'<<2*real(T(t1,t2)*rho(t2,t1));
//	cout<<endl;
//}
