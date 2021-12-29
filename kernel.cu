
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mat.h>
#include <stdio.h>


const int threadsPerBlock=32;
const int blocksPerGrid=8;
#define NUM_TOL 5
#define DIM 50


__device__ void insert_idx(int* index,int &length,int id,int Dim,double* C_cur,double* C_ori,double* inv_CTC,int num_tol,double *tmp,double *D)
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
		C_cur[length*Dim+j]=C_ori[id*Dim+j];
		C_cur[temp*Dim+j]=temp2;
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
	inv_CTC[(length+1)*(length+1)-1]=0.0;
	for(int i=0;i<length;i++)
		inv_CTC[(length+1)*(length+1)-1]+=D[i]*tmp[i];
	inv_CTC[(length+1)*(length+1)-1]=1.0/(B-inv_CTC[(length+1)*(length+1)-1]);

	for(int i=0;i<length;i++)
	{
		inv_CTC[length*num_tol+i]=-tmp[i]*inv_CTC[(length+1)*(length+1)-1];
		inv_CTC[i*num_tol+length]=inv_CTC[length*num_tol+i];
	}
	for(int i=0;i<length;i++)
	{
		for(int j=0;j<=i;j++)
		{
			inv_CTC[i*num_tol+j]+=tmp[i]*tmp[j]*inv_CTC[(length+1)*(length+1)-1];
			inv_CTC[i*num_tol+j]=inv_CTC[j*num_tol+i];
		}
	}
	length++;
}

__device__ void delete_idx(int* index,int &length,int idx,int Dim,double* C_cur,double*C_ori,double* inv_CTC,int num_tol,double *tmp)
{
	int temp=index[idx];
	for(int i=idx;i<length-1;i++)
	{
		index[i]=index[i+1];
		for(int j=0;j<Dim;j++)
			C_cur[i*Dim+j]=C_cur[(i+1)*Dim+j];
	}
	index[length-1]=temp;
	for(int j=0;j<Dim;j++)
		C_cur[(length-1)*Dim+j]=C_ori[temp*Dim+j];

	double den=inv_CTC[idx*num_tol+idx];
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
	for(int i=0;i<idx;i++)
		tmp[i]=inv_CTC[idx*num_tol+i];
	for(int i=idx;i<length-1;i++)
		tmp[i]=inv_CTC[idx*num_tol+i+1];
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


__device__ double maxmin_vec(double* WZ, int length,int num_tol,double &maxval,int &id_max)
{
	double minval=maxval=WZ[length];
	id_max=0;
	for(int i=length+1;i<num_tol;i++)
	{	
		if(WZ[i]>maxval)
		{
			maxval=WZ[i];
			id_max=i;
		}
		else if(WZ[i]<minval)
			minval=WZ[i];
	}
	return minval;
}

__device__ bool anyZPsmallzeros(double* ZP,int length,int *index,double *x,double &alpha)
{
	bool out=false;
	int t=0;
	alpha=1.0e10;
	for(int i=0;i<length;i++)
	{
		if(ZP[i]<=0)
		{
			out=true;
			double temp=(x[i]/(x[i]-ZP[i]));
			if(temp<alpha)
				alpha=temp;
		}			
	}
	return out;
}

__device__ double fnnls(double* C_ori,double* L,int Dim,int num_tol,double* x,double tol,double *W,int *index,double *C_cur,double *inv_CTC,double *tmp,double *z,double *L_re)
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
			W[i]+=L_re[i]*C_ori[i*Dim+j];
		}
	}
	int out_iter=0;
	int iter=0;
	int maxiter=3*num_tol;
	int num=0;
	double maxval;
	int id_max;
	double alpha;
	while(((num_tol-length)!=0)&&maxmin_vec(W,length,num_tol,maxval,id_max)>tol)
	{
		out_iter++;
		for(int i=0;i<num_tol;i++)
			z[i]=0;
		insert_idx(index,length,index[id_max],Dim,C_cur,C_ori,inv_CTC,num_tol,tmp,W);
		ls_seq(C_cur,length,Dim,inv_CTC,num_tol,L,z,tmp);
		while(anyZPsmallzeros(z,length,index,x,alpha))
		{
			iter++;
			if (iter>maxiter)
			{
				for(int i=0;i<num_tol;i++)
					x[index[i]]=z[index[i]];
				double out=0.0;
				for(int i=0;i<Dim;i++)
					out+=(L_re[i]*L_re[i]);
				return out;
			}
			for(int i=0;i<num_tol;i++)
			{
				x[index[i]]=x[index[i]]+alpha*(z[i]-x[index[i]]);
				if(x[index[i]]<tol)
					delete_idx(index,length,i,Dim,C_cur,C_ori,inv_CTC,num_tol,tmp);				
			}
			for(int i=0;i<num_tol;i++)
				z[i]=0;
			ls_seq(C_cur,length,Dim,inv_CTC,num_tol,L,z,tmp);
		}
		
		for(int i=0;i<num_tol;i++)
			x[i]=z[i];
		for(int i=0;i<Dim;i++)
		{
			L_re[i]=0;
			for(int j=0;j<num_tol;j++)
				L_re[i]+=C_cur[j*Dim+i]*x[j];
			L_re[i]=(L[i]-L_re[i]);
		}
		for(int i=0;i<num_tol;i++)
		{
			W[i]=0.0;
			for(int j=0;j<Dim;j++)
			{
				W[i]+=L_re[i]*C_cur[i*Dim+j];
			}
		}
	}
	for(int i=0;i<num_tol;i++)
		x[index[i]]=z[index[i]];
	double out=0.0;
	for(int i=0;i<Dim;i++)
		out+=(L_re[i]*L_re[i]);
	return out;
}
__global__ void fnnls_GPU(double* C_ori,double *C_cur,double *inv_CTC,double* L,int Dim,int num_tol,int num_L,double* x,double *res,double tol)
{	
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	double tmp[NUM_TOL],W[NUM_TOL],z[NUM_TOL],L_re[DIM];
	int index[NUM_TOL];
	for(int i=0;i<NUM_TOL;i++)
		index[i]=0;
	while(tid<num_L)
	{
		res[tid]=fnnls(C_ori,L+tid*Dim,Dim,num_tol,x+tid*num_tol,tol,W,index,C_cur+tid*Dim*num_tol,inv_CTC+tid*num_tol*num_tol,tmp,z,L_re);
		tid+=blockDim.x*gridDim.x;
	}

}

int main()
{
	MATFile *pmatFile1=NULL;
	mxArray *pMxArray1=NULL;

	pmatFile1=matOpen("cup95eff_cube.mat","r");
	int var_num;
	char **varname=matGetDir(pmatFile1, &var_num);
	double *HSI;
	const size_t *Dim_info,*Dim_info2;
	int H,W,Dim;
	pMxArray1=matGetVariable(pmatFile1,varname[0]);   
	Dim_info=mxGetDimensions(pMxArray1);
	H=Dim_info[0];
	W=Dim_info[1];
	Dim=Dim_info[2];

	double *HSI_re=new double[H*W*Dim];
	for(int i=0;i<H*W;i++)
	{
		for(int j=0;j<Dim;j++)
			HSI_re[i*Dim+j]=HSI[j*H*W+i];
	}
	mxFree(HSI);
	double *d_HSI;
    cudaMalloc((void**)&d_HSI,W*H*Dim*sizeof(double));
	cudaMemcpy(d_HSI,HSI_re,Dim*H*W*sizeof(double),cudaMemcpyHostToDevice);

	MATFile *pmatFile2=NULL;
	mxArray *pMxArray2=NULL;
	pmatFile2=matOpen("Endmember.mat","r");
	char **varname2=matGetDir(pmatFile2, &var_num);
	double *Endmember;
	pMxArray2=matGetVariable(pmatFile2,varname2[0]);
	Dim_info2=mxGetDimensions(pMxArray2);
	int num_tol=Dim_info2[1];
	Endmember=(double*)mxGetData(pMxArray2);
	double *d_endmember;
    cudaMalloc((void**)&d_endmember,num_tol*Dim*sizeof(double));
	cudaMemcpy(d_endmember,Endmember,num_tol*Dim*sizeof(double),cudaMemcpyHostToDevice);
    mxFree(Endmember);

	int thres=blocksPerGrid*threadsPerBlock;

	double *d_C_cur;
    cudaMalloc((void**)&d_C_cur,num_tol*Dim*thres*sizeof(double));
	double *d_inv_CTC;
    cudaMalloc((void**)&d_inv_CTC,num_tol*num_tol*thres*sizeof(double));
	double *d_x;
    cudaMalloc((void**)&d_x,H*W*num_tol*sizeof(double));
	double *d_res;
    cudaMalloc((void**)&d_res,H*W*sizeof(double));
	fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_endmember,d_C_cur,d_inv_CTC, d_HSI,Dim,num_tol,H*W,d_x,d_res,1.0e-6);

	
	double* x=new double[H*W*num_tol];
	double* res=new double[H*W];
	cudaMemcpy(x,d_x,sizeof(double)*H*W*num_tol,cudaMemcpyDeviceToHost);
	cudaMemcpy(res,d_res,sizeof(double)*H*W,cudaMemcpyDeviceToHost);

	delete[]x;x=NULL;
	delete[]res;res=NULL;
	cudaFree(d_HSI);
	cudaFree(d_endmember);
	cudaFree(d_C_cur);
	cudaFree(d_inv_CTC);
	cudaFree(d_x);
	cudaFree(d_res);
    return 0;
}

