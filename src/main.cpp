#include <iostream>
#include <fstream>
#include "counter.h"
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
	double gamma = 0.003, Temp = 0.03, omega = 0.003, Ed = 0.0, g0 = 0.005;
	//
	double ek0 = 1e-3, ek1 = 1e-1;
	// TODO: for now impose start from no population on impurities. need to decide which state to start if having initila population
	// TODO: to avoid recurrence of the bath (which is around 2*pi*dos?), need ek > 1e-1, need to think about how to avoid this
	int nek = 60, state = 0;
	int sample = 10000;
	//
	double time0 = 0.0, time1 = 20000.0;
	int ntime = 400;
	//
	if ( rank == 0 )
		cin>>omega>>Ed>>g0>>gamma>>Temp>>ek0>>ek1>>nek>>sample>>time0>>time1>>ntime;
	MPI_Bcast(&omega  ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Ed     ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&g0     ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&gamma  ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&Temp   ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek0    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ek1    ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&nek    ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&sample ,1,MPI_INT,0,MPI_COMM_WORLD);
	MPI_Bcast(&time0  ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&time1  ,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Bcast(&ntime  ,1,MPI_INT,0,MPI_COMM_WORLD);
	//====================
	potential HH;
	ionic AA;
	electronic EE;
	counter time_evo;
	vec x = linspace(-12,12,240);
	mat rho0(HH.sz_s,HH.sz_s,fill::zeros);
	//
	double mass = 1/omega;
	double beta = 1/Temp;
	double xstart = 0.0;
	double xend = 11.8;
	//double sigma_x = 0.5;
	//
	int sample_myself;
	double tmp,tmp2;
	double ave_hops=0.0;
	//
	vec vv = linspace(sqrt(2*ek0/mass),sqrt(2*ek1/mass),nek);
	vec counter_t(HH.sz_f,fill::zeros), counter_r(HH.sz_f,fill::zeros);
	mat mytraj(x.n_rows,HH.sz_f,fill::zeros);
	mat tottraj(x.n_rows,HH.sz_f,fill::zeros);
	vec myhop(x.n_rows,fill::zeros);
	vec tothop(x.n_rows,fill::zeros);
	//
	sample_myself = sample / size;
	//
	HH.generate_H(x,omega,g0,Ed,gamma);
	//past = clock::now();
	HH.diag_H();
	//now = clock::now();
	//if (rank == 0) cout<<"time for diag is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl<<endl;
	for (int iv = 0; iv<nek; iv++)
	{
		time_evo.init(ntime,time0,time1);
		counter_t = counter_t*0;
		counter_r = counter_r*0;
		ave_hops = 0;
		mytraj.zeros();
		tottraj.zeros();
		myhop.zeros();
		tothop.zeros();
		for (int isample=0; isample<sample_myself;isample++)
		{
			//TODO: no rand for testing
			//AA.init(HH,mass,vv(iv)+randn()*(0.5/sigma_x)/mass,xstart+randn()*sigma_x,state,-xend,xend);
			AA.init(HH,mass,vv(iv),xstart,state,-xend,xend);
			EE.init_rho(rho0,HH,beta);
			// run fssh
			double tmp_time = 0.0;
			for (int t1=0; t1<48000; t1++)
			{
				AA.move(HH);
				mytraj(AA.ind_pre,AA.istate) += 1.0;
				myhop(AA.ind_pre) += AA.nhops;
	//past = clock::now();
				EE.evolve(HH,AA);
	//now = clock::now();
	//if (rank == t1%size) cout<<rank<<":  time for evolve is "<<(now.time_since_epoch().count() - past.time_since_epoch().count())/1e9<<'s'<<endl;
				//cout<<t1*AA.dt<<'\t';
				EE.fit_drho(HH,AA,3);
				//EE.try_decoherence(AA);
				AA.try_hop(HH,EE.rho_fock_old,EE.hop_bath);
				tmp_time += AA.dt;
				time_evo.add(tmp_time,AA.ek,AA.etot,AA.istate);
				if (tmp_time > time1)
					break;
				if (abs(AA.check_stop()))
					break;
			}
			// count rate
			if (x(AA.ind_new) < 0)
				counter_r(AA.istate) += 1.0;
			else
				counter_t(AA.istate) += 1.0;
			ave_hops += AA.nhops;
		}
		// collect counting
		for (int t1=0; t1<HH.sz_f; t1++)
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
		//
		MPI_Allreduce(time_evo.ek,time_evo.ek_all,time_evo.nbin,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(time_evo.et,time_evo.et_all,time_evo.nbin,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(time_evo.p0,time_evo.p0_all,time_evo.nbin,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(time_evo.p1,time_evo.p1_all,time_evo.nbin,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		MPI_Allreduce(time_evo.count,time_evo.count_all,time_evo.nbin,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
		//
		double* tmphop = new double[x.n_rows];
		double* tmphopsum = new double[x.n_rows];
		for (size_t t2=0; t2<x.n_rows; t2++)
			tmphop[t2] = myhop(t2);
		MPI_Allreduce(tmphop,tmphopsum,x.n_rows,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		for (size_t t2=0; t2<x.n_rows; t2++)
			tothop(t2) = tmphopsum[t2];
		//
		tmp = ave_hops;
		MPI_Allreduce(&tmp,&tmp2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
		ave_hops = tmp2;
		// print
		if (rank == 0)
		{
			ofstream ff;
			//ff.open("traj_"+to_string(iv)+".dat");
			//for(size_t t1=0; t1<x.n_rows; t1++)
			//	if (x(t1) >= xstart && x(t1) <= xend)
			//	{
			//		ff<<(x(t1)-xstart)/vv(iv);
			//		for(int t2=0; t2<HH.sz_f; t2++)
			//			ff<<'\t'<<tottraj(t1,t2)/sample_myself/size;
			//		ff<<endl;
			//	}
			//ff.close();
			//
			ff.open("m_v-"+to_string(iv)+".dat");
			for(int t1=0; t1<time_evo.nbin; t1++)
				ff<<(time_evo.dt[t1]+time_evo.dt[t1+1])/2<<'\t'<<time_evo.ek_all[t1]/time_evo.count_all[t1]<<'\t'<<time_evo.et_all[t1]/time_evo.count_all[t1]<<'\t'<<time_evo.count_all[t1]<<endl;
			ff.close();
			//
			ff.open("traj_v-"+to_string(iv)+".dat");
			for(int t1=0; t1<time_evo.nbin; t1++)
				ff<<(time_evo.dt[t1]+time_evo.dt[t1+1])/2<<'\t'<<time_evo.p0_all[t1]/time_evo.count_all[t1]<<'\t'<<time_evo.p1_all[t1]/time_evo.count_all[t1]<<endl;
			ff.close();
			//
			//ff.open("hop_v-"+to_string(iv)+".dat");
			//for(size_t t1=0; t1<x.n_rows; t1++)
			//	if (x(t1) >= xstart && x(t1) <= xend)
			//		ff<<(x(t1)-xstart)/vv(iv)<<'\t'<<tothop(t1)/sample_myself/size<<endl;
			//ff.close();
			//
			cout<<vv(iv)*vv(iv)*mass/2;
			for (int t1=0; t1<HH.sz_f; t1++)
				cout<<'\t'<<counter_t(t1)/sample_myself/size;
			for (int t1=0; t1<HH.sz_f; t1++)
				cout<<'\t'<<counter_r(t1)/sample_myself/size;
			//cout<<endl;
			cout<<'\t'<<ave_hops/sample_myself/size<<endl;
		}
	}
	MPI_Finalize();
	return 0;
}
