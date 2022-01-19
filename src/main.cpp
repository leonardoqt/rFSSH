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
	//
	double E1 = -0.1, E2 = -0.11, vdd = 0.04;
	double gamma1 = 0.02, gamma2 = 0.0;
	double temperature = 0.1;
	double beta = 1/temperature;
	//
	double ek0 = 1e-3, ek1 = 1e-1;
	// TODO: for now impose start from no population on impurities. need to decide which state to start if having initila population
	// TODO: to avoid recurrence of the bath, need ek > 1e-1, need to think about how to avoid this
	int nek = 60, state = 0;
	int sample = 10000;
	//
	if ( rank == 0 )
		cin>>E1>>E2>>vdd>>gamma1>>gamma2>>ek0>>ek1>>nek>>sample;
	MPI_Bcast(&E1     ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&E2     ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&vdd    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&gamma1 ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&gamma2 ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek0    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek1    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&nek    ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&sample ,1,MPI_INT,0,MPI_COMM_WORLD);
	//MPI_Bcast(&state  ,1,MPI_INT,0,MPI_COMM_WORLD);
	//====================
	potential HH;
	ionic AA;
	electronic EE;
	vec x = linspace(-8,8,1600);
	mat rho0(HH.sz_s,HH.sz_s,fill::zeros);
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
	vec counter_t(HH.sz_fock,fill::zeros), counter_r(HH.sz_fock,fill::zeros);
	//
	sample_myself = sample / size;
	//
	HH.generate_H(x,E1,E2,vdd,gamma1,gamma2);
	HH.diag_H();
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
			EE.init_rho(rho0,HH,beta);
			// run fssh
			for (int t1=0; t1<24000; t1++)
			{
				AA.move(HH);
				EE.evolve(HH,AA);
				EE.fit_drho_v2(HH,AA);
				//EE.try_decoherence(AA);
				AA.try_hop(HH,EE.rho_fock_old,EE.hop_bath);
				if (abs(AA.check_stop()))
					break;
			}
			// count rate
			if (x(AA.ind_new) < 0)
				counter_r(AA.istate) += 1.0;
			else
				counter_t(AA.istate) += 1.0;
		}
		// collect counting
		for (int t1=0; t1<HH.sz_fock; t1++)
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
			for (int t1=0; t1<HH.sz_fock; t1++)
				cout<<'\t'<<counter_t(t1)/sample_myself/size;
			for (int t1=0; t1<HH.sz_fock; t1++)
				cout<<'\t'<<counter_r(t1)/sample_myself/size;
			cout<<endl;
			//cout<<'\t'<<ave_hops/sample_myself/size<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
