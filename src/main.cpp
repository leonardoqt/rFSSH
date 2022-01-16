#include <iostream>
#include "potential.h"
#include "ionic.h"
#include "electronic.h"
#include <mpi.h>
#include <time.h>

using namespace std;
using namespace arma;

int main()
{
	MPI_Init(NULL,NULL);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	arma_rng::set_seed(time(0)+rank*10);
	//double dE, v0, v1, v2, bb, ek0, ek1;
	//int opt_option=0;
	double B0=3e-3,beta=1e2;
	double ek0, ek1;
	int nek = 60, state = 0;
	int sample = 10000;
	if ( rank == 0 )
		cin>>B0>>beta>>ek0>>ek1>>nek>>sample>>state;
	MPI_Bcast(&B0     ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&beta   ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek0    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek1    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&nek    ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&sample ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&state  ,1,MPI_INT,0,MPI_COMM_WORLD);
	//====================
	potential HH;
	ionic AA;
	electronic EE;
	vec x = linspace(-8,8,3200);
	vec psi0(HH.dim,fill::zeros);
	//
	double mass = 2000;
	double xstart = -6;
	double xend = 6;
	double sigma_x = 0.5;
	//
	int sample_myself;
	double tmp,tmp2;
	int tmpi, tmpi2;
	double ave_hops=0.0;
	//
	vec vv = linspace(sqrt(2*ek0/mass),sqrt(2*ek1/mass),nek);
	vec counter_t(HH.dim,fill::zeros), counter_r(HH.dim,fill::zeros);
	//
	sample_myself = sample / size;
	psi0(state) = 1;
	//
	HH.generate_H2_with_bath(x,beta,B0);
	HH.diag_HB_with_bath();
	////
	//if (rank ==0)
	//	for(uword t1=0; t1< x.n_elem; t1++)
	//		cout<<x(t1)<<'\t'<<HH.eigval.col(t1).t();
	//MPI_Finalize();
	//return 0;
	////
	for (int iv = 0; iv<nek; iv++)
	{
		counter_t = counter_t*0;
		counter_r = counter_r*0;
		for (int t0=0; t0<sample_myself;t0++)
		{
			AA.init(HH,mass,vv(iv)+randn()*(0.5/sigma_x)/mass,xstart+randn()*sigma_x,state,-xend,xend);
			EE.init_psi(psi0);
			// run fssh
			for (int t1=0; t1<24000; t1++)
			{
				AA.move_with_bath(HH);
				EE.evolve_with_bath_v2(HH,AA);
				//EE.try_decoherence(AA);
				AA.try_hop_with_bath_state2(HH,EE.rho);
				//AA.print_rate(x,HH,EE.rho);
				//cout<<x(AA.ind_new)<<'\t'<<AA.v_new<<endl;
				if (abs(AA.check_stop()))
					break;
			}
			// TODO istate changes one the right side of the potential when B increases
			// perhaps need to align them?
			// count rate
			if (x(AA.ind_new) < 0)
				counter_r(AA.istate) += 1.0;
			else
				counter_t(AA.istate) += 1.0;
		}
		// collect counting
		for (int t1=0; t1<HH.dim; t1++)
		{
			tmp = counter_t(t1);
			MPI_Allreduce(&tmp,&tmp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			counter_t(t1) = tmp2;
			tmp = counter_r(t1);
			MPI_Allreduce(&tmp,&tmp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			counter_r(t1) = tmp2;
		}
		tmpi = AA.nhops;
		MPI_Allreduce(&tmpi,&tmpi2,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		ave_hops = ave_hops + tmpi2;
		// print
		if (rank == 0)
		{
			cout<<vv(iv)*vv(iv)*mass/2;
			for (int t1=0; t1<HH.dim; t1++)
				cout<<'\t'<<counter_t(t1)/sample_myself/size;
			for (int t1=0; t1<HH.dim; t1++)
				cout<<'\t'<<counter_r(t1)/sample_myself/size;
			cout<<'\t'<<ave_hops/sample_myself/size<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
