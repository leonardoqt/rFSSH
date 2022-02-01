#include <iostream>
#include <fstream>
#include "potential.h"
#include "ionic.h"
#include "electronic.h"
#include <mpi.h>
#include <time.h>
#include <chrono>

using namespace std;
using namespace arma;

int main()
{
	typedef chrono::high_resolution_clock clock;
	typedef chrono::high_resolution_clock::time_point timepoint;
	//timepoint past, loop0, loop1, run0, run1;
	timepoint now = clock::now();
	MPI_Init(NULL,NULL);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	arma_rng::set_seed(now.time_since_epoch().count()+rank*10);
	//
	double E1 = -0.1, E2 = -0.11, vdd = 0.04;
	double gamma1 = 0.02, gamma2 = 0.0;
	double temperature = 0.1;
	double beta = 1/temperature;
	//
	double ek0 = 1e-3, ek1 = 1e-1;
	// TODO: for now impose start from no population on impurities. need to decide which state to start if having initila population
	// TODO: to avoid recurrence of the bath (which is around 2*pi*dos?), need ek > 1e-1, need to think about how to avoid this
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
	//double sigma_x = 0.5;
	//
	int sample_myself;
	double tmp,tmp2;
	int tmpi, tmpi2;
	double ave_hops=0.0;
	//
	vec vv = linspace(sqrt(2*ek0/mass),sqrt(2*ek1/mass),nek);
	vec counter_t(HH.sz_fock,fill::zeros), counter_r(HH.sz_fock,fill::zeros);
	mat mytraj(x.n_rows,HH.sz_fock,fill::zeros);
	mat tottraj(x.n_rows,HH.sz_fock,fill::zeros);
	vec myhop(x.n_rows,fill::zeros);
	vec tothop(x.n_rows,fill::zeros);
	//
	sample_myself = sample / size;
	//
	//if (rank == 0) cout<<E1<<' '<<E2<<' '<<vdd<<' '<<gamma1<<' '<<gamma2<<' '<<ek0<<' '<<ek1<<' '<<nek<<' '<<sample<<endl;
	//past = clock::now();
	HH.generate_H(x,E1,E2,vdd,gamma1,gamma2);
	//now = clock::now();
	//if (rank == 0) cout<<"time for gen is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
	//past = clock::now();
	HH.diag_H();
	//now = clock::now();
	//if (rank == 0) cout<<"time for diag is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl<<endl;
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
		ave_hops = 0;
		mytraj.zeros();
		tottraj.zeros();
		myhop.zeros();
		tothop.zeros();
	//run0 = clock::now();
		for (int isample=0; isample<sample_myself;isample++)
		{
			//TODO: no rand for testing
			//AA.init(HH,mass,vv(iv)+randn()*(0.5/sigma_x)/mass,xstart+randn()*sigma_x,state,-xend,xend);
			AA.init(HH,mass,vv(iv),xstart,state,-xend,xend);
			EE.init_rho(rho0,HH,beta);
			// run fssh
			for (int t1=0; t1<2400; t1++)
			{
	//loop0 = clock::now();
	//past = clock::now();
				AA.move(HH);
				mytraj(AA.ind_pre,AA.istate) += 1.0;
				myhop(AA.ind_pre) = AA.nhops;
	//now = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for move is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
	//past = clock::now();
				EE.evolve(HH,AA);
	//now = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for evolve is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
	//past = clock::now();
				//cout<<t1*AA.dt<<'\t';
				EE.fit_drho_v1(HH,AA);
	//now = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for fit is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
	//past = clock::now();
				//EE.try_decoherence(AA);
				AA.try_hop(HH,EE.rho_fock_old,EE.hop_bath);
	//now = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for hop is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
				if (abs(AA.check_stop()))
					break;
	//loop1 = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for one loop is "<<(loop1.time_since_epoch().count() - loop0.time_since_epoch().count())/1e9<<'s'<<"   "<<AA.ind_new<<endl<<endl;
			}
			// count rate
			if (x(AA.ind_new) < 0)
				counter_r(AA.istate) += 1.0;
			else
				counter_t(AA.istate) += 1.0;
		}
	//run1 = clock::now();
	//if (rank == 0) cout<<rank<<":  time for one traj. is "<<(run1.time_since_epoch().count() - run0.time_since_epoch().count())/1e9<<'s'<<"   "<<AA.ind_new<<endl<<endl;
		// collect counting
		for (int t1=0; t1<HH.sz_fock; t1++)
		{
			double* tmptraj = new double[x.n_rows];
			double* tmptrajsum = new double[x.n_rows];
			for (size_t t2=0; t2<x.n_rows; t2++)
				tmptraj[t2] = mytraj(t2,t1);
			MPI_Allreduce(tmptraj,tmptrajsum,x.n_rows,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			for (size_t t2=0; t2<x.n_rows; t2++)
				tottraj(t2,t1) = tmptrajsum[t2];
			delete[] tmptraj;
			delete[] tmptrajsum;
			//
			tmp = counter_t(t1);
			MPI_Allreduce(&tmp,&tmp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			counter_t(t1) = tmp2;
			tmp = counter_r(t1);
			MPI_Allreduce(&tmp,&tmp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
			counter_r(t1) = tmp2;
		}
		double* tmphop = new double[x.n_rows];
		double* tmphopsum = new double[x.n_rows];
		for (size_t t2=0; t2<x.n_rows; t2++)
			tmphop[t2] = myhop(t2);
		MPI_Allreduce(tmphop,tmphopsum,x.n_rows,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		for (size_t t2=0; t2<x.n_rows; t2++)
			tothop(t2) = tmphopsum[t2];
		//
		tmpi = AA.nhops;
		MPI_Allreduce(&tmpi,&tmpi2,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		ave_hops = ave_hops + tmpi2;
		// print
		if (rank == 0)
		{
			ofstream ff;
			ff.open("traj_"+to_string(iv)+".dat");
			for(size_t t1=0; t1<x.n_rows; t1++)
				if (x(t1) >= xstart && x(t1) <= xend)
				{
					ff<<(x(t1)-xstart)/vv(iv);
					for(int t2=0; t2<HH.sz_fock; t2++)
						ff<<'\t'<<tottraj(t1,t2)/sample_myself/size;
					ff<<endl;
				}
			ff.close();
			//
			ff.open("hop_"+to_string(iv)+".dat");
			for(size_t t1=0; t1<x.n_rows; t1++)
				if (x(t1) >= xstart && x(t1) <= xend)
					ff<<(x(t1)-xstart)/vv(iv)<<'\t'<<tothop(t1)/sample_myself/size<<endl;
			ff.close();
			//
			cout<<vv(iv)*vv(iv)*mass/2;
			for (int t1=0; t1<HH.sz_fock; t1++)
				cout<<'\t'<<counter_t(t1)/sample_myself/size;
			for (int t1=0; t1<HH.sz_fock; t1++)
				cout<<'\t'<<counter_r(t1)/sample_myself/size;
			//cout<<endl;
			cout<<'\t'<<ave_hops/sample_myself/size<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
