#include<stdio.h>
#include <mat.h>
#include "string.h"
#include "f2c.h"
#include <vector>

//#include  <cula_lapack.h>
//#include  <cula_lapack_device.h>
//#include <cula.h>
//#include <cula_blas.h>
//#include "cula_blas_device.h"
//#include <cula_types.h>
//#include <cublas_v2.h>

using namespace std;
#define MIN_FLOAT -1e10
#define PI 3.14159265358979323846264338327950288419716939937510

extern"C"
{
#include <clapack.h>
} 

double* leastsquare(double* A, int m,int n,double*B)
{
	char TRANS= 'N';
	integer M = m; 
	integer N = n; 
	integer NRHS=1;
	double* A_b=new double[m*n];
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			A_b[i*m+j]=A[i*m+j];
	integer LDA=m;
	double* B_b=new double[m];
	integer LDB=m;
	for(int i=0;i<m;i++)
		B_b[i]=B[i];
	integer LWORK=4*m;
	double* WORK=new double[LWORK];
	integer INFO;
	dgels_(&TRANS,&M,&N,&NRHS,A_b,&LDA,B_b,&LDB,WORK,&LWORK,&INFO);
	
	delete[]A_b;A_b=NULL;
	delete[]WORK;WORK=NULL;
	double* out=new double[n];
	for(int i=0;i<n;i++)
		out[i]=B_b[i];
	delete[]B_b;B_b=NULL;
	return out;
}

double leastsquare_matrix(double* A, int m,int n,int c,double*B,double *out)
{
	char TRANS= 'N';
	integer M = m; 
	integer N = n; 
	integer NRHS=c;
	double* A_b=new double[m*n];
	for(int i=0;i<m*n;i++)
		A_b[i]=A[i];
	integer LDA=m;
	double* B_b=new double[m*c];
	integer LDB=m;
	for(int i=0;i<m*c;i++)
		B_b[i]=B[i];
	integer LWORK=max(1,min(m,n)+max(min(m,n),NRHS)*3);;
	double* WORK=new double[LWORK];
	integer INFO;
	dgels_(&TRANS,&M,&N,&NRHS,A_b,&LDA,B_b,&LDB,WORK,&LWORK,&INFO);

	delete[]A_b;A_b=NULL;
	delete[]WORK;WORK=NULL;
	double square_res=0.0;
	for(int i=0;i<c;i++)
	{
		for(int j=0;j<n;j++)
		   out[i*n+j]=B_b[i*m+j];
		double tmp=0.0;
		for(int j=n;j<m;j++)
			tmp+=B_b[i*m+j]*B_b[i*m+j];
		square_res+=sqrt(tmp/m);
	}
	delete[]B_b;B_b=NULL;
	return square_res/c;
}
bool anyZPsmallzeros(double* ZP,bool*P,int n,double*x, double&alpha)
{
	bool out=false;
	int t=0;
	alpha=1.0e10;
	for(int i=0;i<n;i++)
	{
		if(P[i])
		{
			if(ZP[t]<=0)
			{
				out=true;
				double temp=(x[i]/(x[i]-ZP[t]));
				if(temp<alpha)
					alpha=temp;
			}
			t++;			
		}
	}
	return out;
}
bool anyboolvec(bool* Z,int n)
{
	bool out=false;
	for(int i=0;i<n;i++)
	{
		if(Z[i])
		{
			out=true;
			break;
		}
	}
	return out;
}
double max_vec(double* Z,bool* R, int n)
{
	double out=MIN_FLOAT;
	for(int i=0;i<n;i++)
	{
		if(R[i])
		{
			if(Z[i]>out)
			{
				out=Z[i];
			}
		}
		
	}
	return out;
}
double fnnls(double* A,double* b,int m,int n,double* x,double tol)
{
	bool* P=new bool[n];
	bool* R=new bool[n];
	double* wz=new double[n];
	double* w=new double[n];
	double* x_temp=new double[n];
	double* resid=new double[m];
	double* APTAP=new double[m*m];
	for(int i=0;i<n;i++)
	{
		P[i]=false;
		R[i]=true;
		x[i]=0;
		wz[i]=0;

	}
	for(int i=0;i<m;i++)
	{
		resid[i]=b[i];
	}
	for(int i=0;i<n;i++)
	{
		w[i]=0;
		for(int j=0;j<m;j++)
		{
			w[i]+=resid[j]*A[i*m+j];
		}
	}
	int iteration=0;
	int maxiter=3*n;
	int num=0;
	while(anyboolvec(R,n)&&(max_vec(w,R,n)>tol))
	{
				
		memset(x_temp,0,sizeof(double)*n);
		int t=0;
		for(int i=0;i<n;i++)
		{

			if (P[i]==true)
				wz[i]= MIN_FLOAT;
			else 
			{
				wz[i]=w[i];
				if (wz[t]<wz[i])
				{
					t=i;
				}
			}
		}
		P[t]=true;
		R[t]=false;
		num++;
		double* AP=new double[m*num];
		int t2=0;
		for(int i=0;i<n;i++)
		{
			if(P[i]==true)
			{
				for(int j=0;j<m;j++)
					AP[t2*m+j]=A[i*m+j];
				t2++;
			}
		}
		
		double* zP=leastsquare(AP,m,num,b);
		delete[]AP;AP=NULL;
		t2=0;
		for(int i=0;i<n;i++)
		{
			if(P[i]==true)
			{
				x_temp[i]=zP[t2];
				t2++;
			}
		}
		double alphia;
		bool condtion1=anyZPsmallzeros(zP,P,n,x,alphia);
		delete []zP;zP=NULL;
		while(condtion1)
		{
			iteration++;
			if (iteration>maxiter)
			{
				for(int i=0;i<m;i++)
				{
					resid[i]=b[i];
					for(int j=0;j<n;j++)
					{
						resid[i]-=A[j*m+i]*x[j];					
					}
				}
				double out=0.0;
				for(int i=0;i<m;i++)
				{
					out+=(resid[i]*resid[i]);
				}	

				return out;
			}
			for(int i=0;i<n;i++)
			{
				x[i]=x[i]+alphia*(x_temp[i]-x[i]);
				if(P[i])
				{
					if(abs(x[i])<tol)
					{
						P[i]=false;
						R[i]=true;
						num--;
					}
				}
			}
			memset(x_temp,0,sizeof(double)*n);
			double* AP2=new double[m*num];
			int t3=0;
			for(int i=0;i<n;i++)
			{
				if(P[i]==true)
				{
					for(int j=0;j<m;j++)
						AP2[t3*m+j]=A[i*m+j];
					t3++;
				}
			}
			double* zP2=leastsquare(AP2,m,num,b);
			delete[]AP2;AP2=NULL;
			t2=0;
			for(int i=0;i<n;i++)
			{
				if(P[i]==true)
				{
					x_temp[i]=zP2[t2];
					t2++;
				}
			}
			condtion1=anyZPsmallzeros(zP2,P,n,x,alphia);
			delete[]zP2;zP2=NULL;
			
		}
		for(int i=0;i<n;i++)
		{
			x[i]=x_temp[i];
		}
		for(int i=0;i<m;i++)
		{
			resid[i]=b[i];
			for(int j=0;j<n;j++)
			{
				resid[i]-=A[j*m+i]*x[j];					
			}
		}
		for(int i=0;i<n;i++)
		{
			w[i]=0;
			for(int j=0;j<m;j++)
			{
				w[i]+=A[i*m+j]*resid[j];
			}
		}
		if(iteration>maxiter)
			break;
	}
	double resnorm=0.0;
	for(int i=0;i<m;i++)
	{
		resnorm+=(resid[i]*resid[i]);
	}	
	delete []wz;
	delete []w;
	delete []x_temp;
	delete []resid;
	delete []APTAP;
	delete []P;
	delete []R;
	P=NULL;
	R=NULL;
	wz=NULL;
	w=NULL;
	x_temp=NULL;
	resid=NULL;
	APTAP=NULL;
	return resnorm;
}
void inv_A(double* A, int H,double*inv_A)
{
	integer M=H;
	integer N=H;

	for(int i=0;i<H;i++)
	{
		for(int j=0;j<H;j++)
			inv_A[i*H+j]=A[i*H+j];
	}
	integer LDA=H;
	integer *IPIV=new integer[H];
	integer INFO;
	dgetrf_(&M,&N,inv_A,&LDA,IPIV,&INFO);
	integer LWORK=3*H;
	double*WORK=new double[LWORK];
	dgetri_(&M,inv_A,&LDA,IPIV,WORK,&LWORK,&INFO);
	delete[]IPIV;IPIV=NULL;
	delete[]WORK;WORK=NULL;

}
double hyperFcls(double* endmember,int endmember_num,int Dim,double* y,double*abundance)
{
	double *Mbckp=new double[Dim*endmember_num];
	int count=endmember_num;
	int* refer=new int[endmember_num];
	for(int j=0;j<endmember_num;j++)
		refer[j]=j;

	for(int i=0;i<endmember_num;i++)
	{
		for(int j=0;j<Dim;j++)
			Mbckp[i*Dim+j]=endmember[i*Dim+j];
	}
	double* als_hat=new double[count]; 
	bool flag=true;
	while (flag)
	{
		double* inv_UTU=new double[count*count];
		double* UTU=new double[count*count];
		for(int i=0;i<count;i++)
		{
			for(int j=0;j<=i;j++)
			{
				UTU[i*count+j]=0;
				for(int k=0;k<Dim;k++)
					UTU[i*count+j]+=Mbckp[i*Dim+k]*Mbckp[j*Dim+k];
				UTU[j*count+i]=UTU[i*count+j];
			}
		}
		inv_A(UTU,count,inv_UTU);
		double* inv_UTU_U=new double[count*Dim];
		for(int i=0;i<count;i++)
		{
			for(int j=0;j<Dim;j++)
			{
				inv_UTU_U[i*Dim+j]=0;
				for(int k=0;k<count;k++)
					inv_UTU_U[i*Dim+j]+=inv_UTU[i*count+k]*Mbckp[k*Dim+j];
			}
		}
		
		for(int i=0;i<count;i++)
		{
			als_hat[i]=0;
			for(int j=0;j<Dim;j++) 
				als_hat[i]+=inv_UTU_U[i*Dim+j]*y[j];
		}
		double*s=new double[count];
		double tmp=0.0;
		double sum_a=-1.0;
		for(int i=0;i<count;i++)
		{
			s[i]=0;
			for(int j=0;j<count;j++)
				s[i]+=inv_UTU[i*count+j];
			tmp+=s[i];
			sum_a+=als_hat[i];
		}
		delete[]UTU;UTU=NULL;
		delete[]inv_UTU;inv_UTU=NULL;
		delete[]inv_UTU_U;inv_UTU_U=NULL;

		for(int i=0;i<count;i++)
			als_hat[i]-=s[i]*sum_a/tmp;
		flag=false;
		double val=0.0;
		int idx;
		for(int i=0;i<count;i++)
		{
			if(als_hat[i]<0)
			{
				flag=true;
				double temp=abs(als_hat[i]/s[i]);
				if(val<temp)
				{
					val=temp;
					idx=i;
				}
			}
		}
		delete[]s;s=NULL;
	
		if(flag)
		{			
			for(int i=idx+1;i<count;i++)
			{
				for(int j=0;j<Dim;j++)
				{
					Mbckp[(i-1)*Dim+j]=Mbckp[i*Dim+j];
				}
				refer[i-1]=refer[i];

			}
			refer[count-1]=-1;;
			count=count-1;
		}
	}
	delete[]Mbckp;Mbckp=NULL;
	for(int i=0;i<endmember_num;i++)
		abundance[i]=0;
	for(int i=0;i<count;i++)
		abundance[refer[i]]=als_hat[i];
	delete[]als_hat;als_hat=NULL;
	double out=0.0;
	for(int i=0;i<Dim;i++)
	{
		double temp=y[i];
		for(int j=0;j<endmember_num;j++)
			temp-=endmember[j*Dim+i]*abundance[j];
		out+=temp*temp;
	}
	return out;
}

//void cula_inv_A(double* A, int H,double*inv_A)
////void inv_A(double* A, int H,double*inv_A)
//{
//	//integer M=H;
//	//integer N=H;
//	
//	for(int i=0;i<H;i++)
//	{
//		for(int j=0;j<H;j++)
//			inv_A[i*H+j]=A[i*H+j];
//	}
//	//integer LDA=H;
//	culaInt *IPIV=new culaInt[3*H];
//	//integer INFO;
//	//dgetrf_(&M,&N,inv_A,&LDA,IPIV,&INFO);
//	culaStatus s;
//	s=culaInitialize();
//	s=culaDgetrf(H,H,inv_A,H,IPIV);
//	if( s != culaNoError )
//	{ 
//		if( s == culaDataError ) 
//			printf("Data error with code %d, please see LAPACK documentation\n", culaGetErrorInfo()); 
//		else 
//			printf("%s\n", culaGetStatusString(s));
//	}
//
//	//integer LWORK=3*H;
//	//double*WORK=new double[LWORK];
//	//dgetri_(&M,inv_A,&LDA,IPIV,WORK,&LWORK,&INFO);
//	culaDgetri(H,inv_A,H,IPIV);
//	culaShutdown();
//	delete[]IPIV;IPIV=NULL;
//	//delete[]WORK;WORK=NULL;
//
//}