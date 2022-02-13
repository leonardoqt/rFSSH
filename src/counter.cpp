#include "counter.h"

void counter::init(int Nbin, double T0, double T1)
{
	nbin = Nbin;
	// TODO deallocate pointers before assign
	dt = new double[nbin+1];
	ek = new double[nbin];
	et = new double[nbin];
	count = new int[nbin];
	ek_all = new double[nbin];
	et_all = new double[nbin];
	count_all = new int[nbin];
	for (int t1=0; t1<=nbin; t1++)
		dt[t1] = T0+(T1-T0)/nbin * t1;
	for (int t1=0; t1<nbin; t1++)
	{
		ek[t1] = 0;
		et[t1] = 0;
		count[t1] = 0;
	}
	last_t = 0;
	last_i = 0;
}

void counter::add(double timei, double eki, double eti)
{
	if (timei < last_t)
	{
		last_i = 0;
		last_t = dt[last_i];
	}
	//
	for(int t1=last_i; t1<nbin; t1++)
	{
		if (timei < dt[t1+1])
		{
			ek[t1] += eki;
			et[t1] += eti;
			count[t1]++;
			last_t = timei;
			last_i = t1;
			break;
		}
	}
}
