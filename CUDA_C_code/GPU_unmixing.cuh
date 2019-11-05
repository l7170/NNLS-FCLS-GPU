#include "cuda_runtime.h"
#include "device_functions.h"



#define NUM_TOL 3
#define DIM 176
__constant__  double C_ori[DIM*NUM_TOL];
__device__ void insert_idx(int* index,int &length,int id,int Dim,double* C_cur,double* inv_CTC,int num_tol,double *tmp,double *D)
{
	int temp;
	for(int i=length;i<num_tol;i++)
	{
		if(index[i]==id)
		{
			temp=index[length];
			index[length]=id;
			index[i]=temp;
			temp=i;
			break;
		}
	}
	for(int j=0;j<Dim;j++)
	{
		double temp2=C_cur[length*Dim+j];
		C_cur[temp*Dim+j]=temp2;
		C_cur[length*Dim+j]=C_ori[id*Dim+j];
		
	}
	for(int i=0;i<length;i++)
	{
		D[i]=0;
		for(int j=0;j<Dim;j++)
			D[i]+=C_cur[i*Dim+j]*C_ori[id*Dim+j];
	}

	for(int i=0;i<length;i++)
	{
		tmp[i]=0;
		for(int j=0;j<length;j++)
			tmp[i]+=inv_CTC[i*num_tol+j]*D[j];
	}
	double B=0.0;
	for(int i=0;i<Dim;i++)
		B+=C_ori[id*Dim+i]*C_ori[id*Dim+i];
	inv_CTC[length*num_tol+length]=0.0;
	for(int i=0;i<length;i++)
		inv_CTC[length*num_tol+length]+=D[i]*tmp[i];
	inv_CTC[length*num_tol+length]=1.0/(B-inv_CTC[length*num_tol+length]);

	for(int i=0;i<length;i++)
	{
		inv_CTC[length*num_tol+i]=-tmp[i]*inv_CTC[length*num_tol+length];
		inv_CTC[i*num_tol+length]=inv_CTC[length*num_tol+i];
	}
	for(int i=0;i<length;i++)
	{
		for(int j=0;j<=i;j++)
		{
			inv_CTC[i*num_tol+j]+=tmp[i]*tmp[j]*inv_CTC[length*num_tol+length];
			inv_CTC[j*num_tol+i]=inv_CTC[i*num_tol+j];
		}
	}
	length++;
}

__device__  void delete_idx(int* index,int &length,int idx,int Dim,double* C_cur,double* inv_CTC,int num_tol,double *tmp)
{
	int temp=index[idx];  // index of deleting endmember in original endmembers matrix
	//update current endmembers matrix and index
	for(int i=idx;i<length-1;i++)
	{
		index[i]=index[i+1];
		for(int j=0;j<Dim;j++)
			C_cur[i*Dim+j]=C_cur[(i+1)*Dim+j];
	}
	index[length-1]=temp;
	for(int j=0;j<Dim;j++)
		C_cur[(length-1)*Dim+j]=C_ori[temp*Dim+j];
	double den=inv_CTC[idx*num_tol+idx];// compute P3(den) in Eq.6
	// compute P2(tmp) in Eq.6
	for(int i=0;i<idx;i++)
		tmp[i]=inv_CTC[idx*num_tol+i];
	for(int i=idx;i<length-1;i++)
		tmp[i]=inv_CTC[idx*num_tol+i+1];
	// compute P1(inv_CTC) in Eq.6
	for(int i=0;i<idx;i++)
	{
		for(int j=idx;j<length-1;j++)
			inv_CTC[i*num_tol+j]=inv_CTC[i*num_tol+j+1];

	}
	for(int i=idx;i<length-1;i++)
	{
		for(int j=0;j<idx;j++)
			inv_CTC[i*num_tol+j]=inv_CTC[(i+1)*num_tol+j];
	}
	for(int i=idx;i<length-1;i++)
	{
		for(int j=idx;j<length-1;j++)
			inv_CTC[i*num_tol+j]=inv_CTC[(i+1)*num_tol+j+1];
	}
	//update (E_p^t*E_p)^-1 in Eq.6
	for(int i=0;i<length-1;i++)
	{
		for(int j=0;j<length-1;j++)
			inv_CTC[i*num_tol+j]-=(tmp[i]*tmp[j])/den;
	}
	length--;
}

__device__ void ls_seq(double* C_cur,int length,int Dim,double* inv_CTC,int num_tol,double *L,double* abundance,double* tmp)
{
	//abundance(lengthx1)=inv_CTC(lengthxlength)*C_cur'(lengthxDim)*L(Dimx1)
	for(int i=0;i<length;i++)
	{
		tmp[i]=0.0;
		for(int j=0;j<Dim;j++)
			tmp[i]+=C_cur[i*Dim+j]*L[j];
	}

	for(int i=0;i<length;i++)
	{
		abundance[i]=0.0;
		for(int j=0;j<length;j++)
			abundance[i]+=inv_CTC[i*num_tol+j]*tmp[j];
	}
} 


__device__ double maxmin_vec(double* WZ, int* indice,int length,int num_tol,int &id_max)
{
 	double maxval=WZ[indice[length]];
	id_max=length;
	for(int i=length+1;i<num_tol;i++)
	{	
		if(WZ[indice[i]]>maxval)
		{
			maxval=WZ[indice[i]];
			id_max=i;
		}
	}
	return maxval;
}

__device__ bool anyZPsmallzeros(double* ZP,int length,int *index,double *x,double &alpha)
{
	bool out=false;
	alpha=1.0e10;
	for(int i=0;i<length;i++)
	{
		if(ZP[i]<=0)
		{
			out=true;
			double temp=(x[index[i]]/(x[index[i]]-ZP[i]));
			if(temp<alpha)
				alpha=temp;
		}			
	}
	return out;
}

__device__ double fnnls(double* L,int Dim,int num_tol,double* x,double tol,double *W,int *index,double *C_cur,double *inv_CTC,double *tmp,double *z,double *L_re)
{
	int length=0;
	for(int i=0;i<num_tol;i++)
	{
		W[i]=0.0;
		x[i]=0.0;
		index[i]=i;
	}
	for(int i=0;i<Dim;i++)
		L_re[i]=L[i];

	for(int i=0;i<num_tol;i++)
	{
		for(int j=0;j<Dim;j++)
		{
			W[i]+=L_re[j]*C_ori[i*Dim+j];
		}
	}
	int out_iter=0;
	int iter=0;
	int maxiter=3*num_tol;
	
	int id_max;
	double alpha;
 	while(((num_tol-length)!=0)&&maxmin_vec(W,index,length,num_tol,id_max)>tol)
	{
		out_iter++;
		for(int i=0;i<num_tol;i++)
			z[i]=0;
		insert_idx(index,length,index[id_max],Dim,C_cur,inv_CTC,num_tol,tmp,W);
		ls_seq(C_cur,length,Dim,inv_CTC,num_tol,L,z,tmp);
 		while(anyZPsmallzeros(z,length,index,x,alpha))
		{
			iter++;
			if (iter>maxiter)
			{
				for(int i=0;i<num_tol;i++)
					x[index[i]]=z[i];
				double out=0.0;
				for(int i=0;i<Dim;i++)
					out+=(L_re[i]*L_re[i]);
				return out;
			}
			for(int i=0;i<num_tol;i++)
			{
				x[index[i]]=x[index[i]]+alpha*(z[i]-x[index[i]]);						
			}
			for(int i=0;i<num_tol;i++)
			{
				if(x[i]<tol)
				{
					for(int j=0;j<length;j++)
					{
						if(index[j]==i)
						{
							delete_idx(index,length,j,Dim,C_cur,inv_CTC,num_tol,tmp);	
							break;
						}
					}
					
				}
			}
			for(int i=0;i<num_tol;i++)
				z[i]=0;
			ls_seq(C_cur,length,Dim,inv_CTC,num_tol,L,z, tmp);
		}

		for(int i=0;i<num_tol;i++)
			x[index[i]]=z[i];
		for(int i=0;i<Dim;i++)
		{
			L_re[i]=0;
			for(int j=0;j<num_tol;j++)
				L_re[i]+=C_ori[j*Dim+i]*x[j];
			L_re[i]=(L[i]-L_re[i]);
		}
		for(int i=0;i<num_tol;i++)
		{
			W[i]=0.0;
			for(int j=0;j<Dim;j++)
			{
				W[i]+=L_re[j]*C_ori[i*Dim+j];
			}
		}
	}
	for(int i=0;i<num_tol;i++)
		x[index[i]]=z[i];
	double out=0.0;
	for(int i=0;i<Dim;i++)
		out+=(L_re[i]*L_re[i]);
	return out;
}
__global__ void fnnls_GPU(double *C_cur,double *inv_CTC,double* L,int Dim,int num_tol,int num_L,double* x,double *res,double tol)
{
	
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double tmp[NUM_TOL],W[NUM_TOL],z[NUM_TOL],L_re[DIM];
	int index[NUM_TOL];
	for(int i=0;i<NUM_TOL;i++)
		index[i]=0;

	while(tid<num_L)
	{
		res[tid]=fnnls(L+tid*Dim,Dim,num_tol,x+tid*num_tol,tol,W,index,C_cur+tid*Dim*num_tol,inv_CTC+tid*num_tol*num_tol,tmp,z,L_re);
		tid+=blockDim.x*gridDim.x;
	}
	//__syncthreads();
}

__device__ double hyperFcls_(double* L,int Dim,int num_tol,double*x,int *index,double *C_cur,double *inv_CTC,double *tmp,double *z,double *L_re)
{
	//initialization
	bool flag=true;
	int count=num_tol;
	for(int i=0;i<num_tol;i++)
		index[i]=i;// initialization of passive-set
	//main-loop
	while(flag)
	{
		//tmp=E_p*y
		for(int i=0;i<count;i++)
		{
			tmp[i]=0;
			for(int j=0;j<Dim;j++)
				tmp[i]+=C_cur[i*Dim+j]*L[j];
		}
		//compute zp at line 5 in Alg.3 
		for(int i=0;i<count;i++)
		{
			z[i]=0;
			for(int j=0;j<count;j++)
				z[i]+=inv_CTC[i*num_tol+j]*tmp[j];
		}
		double temp=0.0;  // numerator of -lambda at line 8 in Alg.3
		double sum_a=-1.0;// denominator  of -lambda at line 8 in Alg.3
		for(int i=0;i<count;i++)
		{
			tmp[i]=0;
			for(int j=0;j<count;j++)
				tmp[i]+=inv_CTC[i*num_tol+j];
			temp+=tmp[i];
			sum_a+=z[i];
		}
		//re-compute zp at line 8 in Alg.3 
		for(int i=0;i<count;i++)
			z[i]-=tmp[i]*sum_a/temp;
		flag=false;
		double val=0.0;
		int idx;   
		for(int i=0;i<count;i++)
		{
			if(z[i]<0)
			{
				flag=true;
				double temp=abs(z[i]/tmp[i]);//  indicator for removing from passive-set
				if(val<temp)
				{
					val=temp;
					idx=i;
				}
			}
		}
		if(flag)
			delete_idx(index,count,idx,Dim,C_cur,inv_CTC,num_tol,tmp); //delete an endmember and update current endmembers matrix by partitioned matrix inversion
		
	}
	// abundance stroes in x
	for(int i=0;i<num_tol;i++)
		x[i]=0;
	for(int i=0;i<count;i++)
		x[index[i]]=z[i];
	//reconstruction residual error stores in return-value
	double out=0.0;
	for(int i=0;i<Dim;i++)
	{
		L_re[i]=0;
		for(int j=0;j<num_tol;j++)
			L_re[i]+=C_ori[j*Dim+i]*x[j];
		out+=pow(L[i]-L_re[i],2.0);
	}
	return out;
}
__global__ void FCLS_GPU(double *C_cur,double *inv_CTC,double* L,int Dim,int num_tol,int num_L,double* x,double *res)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x; // idx is the identification number of the thread
	double tmp[NUM_TOL],z[NUM_TOL],L_re[DIM];  // temporary variables that stores in the registers of each thread
	int index[NUM_TOL];                        // index of endmember in current endmembers matrix
	while(tid<num_L)
	{
		//Compute reconstruction residual error and abundance by device function
		res[tid]=hyperFcls_(L+tid*Dim,Dim,num_tol,x+tid*num_tol,index,C_cur+tid*Dim*(num_tol),inv_CTC+tid*(num_tol*num_tol),tmp,z,L_re); 
		tid+=blockDim.x*gridDim.x;
	}
}

__device__ double hyperfnnls_(double* L,int Dim,int num_tol,double*x,int *index,double *C_cur,double *inv_CTC,double *tmp,double *z,double *L_re)
{
	bool flag=true;
	int count=num_tol;
	for(int i=0;i<num_tol;i++)
		index[i]=i;
	while(flag)
	{
		for(int i=0;i<count;i++)
		{
			tmp[i]=0;
			for(int j=0;j<Dim;j++)
				tmp[i]+=C_cur[i*Dim+j]*L[j];
		}
		for(int i=0;i<count;i++)
		{
			z[i]=0;
			for(int j=0;j<count;j++)
				z[i]+=inv_CTC[i*num_tol+j]*tmp[j];
		}
		double temp=0.0;
		flag=false;
		double val=0.0;
		int idx;
		for(int i=0;i<count;i++)
		{
			if(z[i]<0)
			{
				flag=true;
				temp=abs(z[i]/tmp[i]);
				if(val<temp)
				{
					val=temp;
					idx=i;
				}
			}
		}
		if(flag)
			delete_idx(index,count,idx,Dim,C_cur,inv_CTC,num_tol,tmp);
		
	}
	for(int i=0;i<num_tol;i++)
		x[i]=0;
	for(int i=0;i<count;i++)
		x[index[i]]=z[i];
	double out=0.0;
	for(int i=0;i<Dim;i++)
	{
		L_re[i]=0;
		for(int j=0;j<num_tol;j++)
			L_re[i]+=C_ori[j*Dim+i]*x[j];
		out+=pow(L[i]-L_re[i],2.0);
	}
	return out;
}
__global__ void e_fnnls_GPU(double *C_cur,double *inv_CTC,double* L,int Dim,int num_tol,int num_L,double* x,double *res)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double tmp[NUM_TOL],z[NUM_TOL],L_re[DIM];
	int index[NUM_TOL];

	while(tid<num_L)
	{
		
		res[tid]=hyperfnnls_(L+tid*Dim,Dim,num_tol,x+tid*num_tol,index,C_cur+tid*Dim*(num_tol),inv_CTC+tid*(num_tol*num_tol),tmp,z,L_re);
		tid+=blockDim.x*gridDim.x;
	}

	//__syncthreads();
}
