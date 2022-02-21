#ifndef __MY_COUNTER__
#define __MY_COUNTER__

class counter;

class counter
{
public:
	int nbin;
	double time0, time1;
	double *dt, *ek, *et, *p0, *p1;
	int *count;
	double *ek_all, *et_all, *p0_all, *p1_all;
	int *count_all;
	double last_t;
	int last_i;
	//
	void init(int Nbin, double T0, double T1);
	void add(double timei, double eki, double eti, int istate);
};

#endif
