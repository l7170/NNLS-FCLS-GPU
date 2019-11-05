
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <mat.h>
#include <stdio.h>
#include "GPU_unmixing.cuh"
#include "nonnegleastsquare_cpu.h"
#include <time.h>




const int threadsPerBlock=64;
const int blocksPerGrid=8;

int main(int argc, char* argv[])
{
	MATFile *pmatFile1=NULL;
	mxArray *pMxArray1=NULL;

	pmatFile1=matOpen(argv[1],"r");
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
	HSI=(double*)mxGetData(pMxArray1);
	double *HSI_re=new double[H*W*Dim];
	for(int i=0;i<H*W;i++)
	{
		for(int j=0;j<Dim;j++)
			HSI_re[i*Dim+j]=HSI[j*H*W+i];
	}
	mxFree(HSI);
	matClose(pmatFile1);

	MATFile *pmatFile2=NULL;
	mxArray *pMxArray2=NULL;
	pmatFile2=matOpen(argv[2],"r");
	char **varname2=matGetDir(pmatFile2, &var_num);
	double *Endmember;
	pMxArray2=matGetVariable(pmatFile2,varname2[0]);
	Dim_info2=mxGetDimensions(pMxArray2);
	int num_tol=Dim_info2[1];
	Endmember=(double*)mxGetData(pMxArray2);
	matClose(pmatFile2);

	int show=H*W-50;//50;//

	
	bool isFullConstraint=true;//false;//
	bool flag2=true;//false;//
	bool flag1=true;
	if(!isFullConstraint)
	{
		//GPU----2
		if(flag1)
		{
			cudaDeviceProp prop;
			int whichDevice;
			cudaGetDevice(&whichDevice);
			cudaGetDeviceProperties(&prop,whichDevice);
			if(!prop.deviceOverlap)
			{
				printf("Device will not handle overlaps, so no \"speed up from stream\"\n.");
				return 0;
			}
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaStream_t stream0,stream1;
			cudaStreamCreate(&stream0);
			cudaStreamCreate(&stream1);

			int num_threads=2048;
			int iter=H*W/(2*num_threads);
			int num_last=(H*W)%(2*num_threads);
			if(num_last==0)
			{
				iter--;
				num_last=2*num_threads;
			}
			int iteration=0;
			int num_last0,num_last1;
			num_last0=num_last1=num_last/2;
			if(num_last%2==1)
			{
				num_last1=num_last0+1;
			}


			double *host_HSI;
			cudaHostAlloc((void**)&host_HSI,W*H*Dim*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<W*H*Dim;i++)
				host_HSI[i]=HSI_re[i];
			double *d_HSI0,*d_HSI1;
			cudaMalloc((void**)&d_HSI0,num_threads*Dim*sizeof(double));
			cudaMalloc((void**)&d_HSI1,num_threads*Dim*sizeof(double));
			cudaMemcpyToSymbol(C_ori,Endmember,sizeof(double)*num_tol*Dim);

			double *d_C_cur0,*d_C_cur1;
			cudaMalloc((void**)&d_C_cur0,(num_tol)*Dim*num_threads*sizeof(double));
			cudaMalloc((void**)&d_C_cur1,(num_tol)*Dim*num_threads*sizeof(double));
			double* host_C_cur;
			cudaHostAlloc((void**)&host_C_cur,num_threads*Dim*num_tol*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*Dim;j++)
					host_C_cur[i*num_tol*Dim+j]=Endmember[j];

			double *d_inv_CTC0,*d_inv_CTC1;
			cudaMalloc((void**)&d_inv_CTC0,(num_tol)*(num_tol)*num_threads*sizeof(double));
			cudaMalloc((void**)&d_inv_CTC1,(num_tol)*(num_tol)*num_threads*sizeof(double));

			double* inv_UTU=new double[num_tol*num_tol];
			double* UTU=new double[num_tol*num_tol];
			for(int i=0;i<num_tol;i++)
			{
				for(int j=0;j<=i;j++)
				{
					UTU[i*num_tol+j]=0;
					for(int k=0;k<Dim;k++)
						UTU[i*num_tol+j]+=Endmember[i*Dim+k]*Endmember[j*Dim+k];
					UTU[j*num_tol+i]=UTU[i*num_tol+j];
				}
			}	
			//cula_inv_A(UTU,num_tol,inv_UTU);//inv_A(UTU,num_tol,inv_UTU);//
			inv_A(UTU,num_tol,inv_UTU);
			double *host_inv_UTU;
			cudaHostAlloc((void**)&host_inv_UTU,num_threads*num_tol*num_tol*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*num_tol;j++)
					host_inv_UTU[i*num_tol*num_tol+j]=inv_UTU[j];
			mxFree(Endmember);	
			delete[]inv_UTU;inv_UTU=NULL;
			delete[]UTU;UTU=NULL;

			double *d_x0,*d_x1;
			cudaMalloc((void**)&d_x0,num_threads*num_tol*sizeof(double));
			cudaMalloc((void**)&d_x1,num_threads*num_tol*sizeof(double));
			double *d_res0,*d_res1;
			cudaMalloc((void**)&d_res0,num_threads*sizeof(double));
			cudaMalloc((void**)&d_res1,num_threads*sizeof(double));
			double* host_x;cudaHostAlloc((void**)&host_x,W*H*num_tol*sizeof(double),cudaHostAllocDefault);
			double* host_res;cudaHostAlloc((void**)&host_res,W*H*sizeof(double),cudaHostAllocDefault);

			cudaEventRecord(start,0);
			
			for(int i=0;i<H*W;i+=(2*num_threads))
			{
				if(iteration!=iter)
				{
					cudaMemcpyAsync(d_HSI0,host_HSI+i*Dim,sizeof(double)*num_threads*Dim,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_C_cur0,host_C_cur,sizeof(double)*num_tol*Dim*num_threads,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_inv_CTC0,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_threads,cudaMemcpyHostToDevice,stream0);

					cudaMemcpyAsync(d_HSI1,host_HSI+(i+num_threads)*Dim,sizeof(double)*num_threads*Dim,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_C_cur1,host_C_cur,sizeof(double)*num_tol*Dim*num_threads,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_inv_CTC1,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_threads,cudaMemcpyHostToDevice,stream1);

					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock,0,stream0>>>(d_C_cur0,d_inv_CTC0,d_HSI0,Dim,num_tol,num_threads,d_x0,d_res0);
					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_C_cur1,d_inv_CTC1,d_HSI1,Dim,num_tol,num_threads,d_x1,d_res1);
				
					cudaMemcpyAsync(host_x+i*num_tol,d_x0,sizeof(double)*num_tol*num_threads,cudaMemcpyDeviceToHost,stream0);
					cudaMemcpyAsync(host_res+i,d_res0,sizeof(double)*num_threads,cudaMemcpyDeviceToHost,stream0);

					cudaMemsetAsync(d_x0,0,num_tol*num_threads*sizeof(double),stream0);
					cudaMemsetAsync(d_res0,0,num_threads*sizeof(double),stream0);
				 
					cudaMemcpyAsync(host_x+(i+num_threads)*num_tol,d_x1,sizeof(double)*num_tol*num_threads,cudaMemcpyDeviceToHost,stream1);
					cudaMemcpyAsync(host_res+(i+num_threads),d_res1,sizeof(double)*num_threads,cudaMemcpyDeviceToHost,stream1);

					cudaMemsetAsync(d_x1,0,num_tol*num_threads*sizeof(double),stream1);
					cudaMemsetAsync(d_res1,0,num_threads*sizeof(double),stream1);
					iteration++;
				}
				else
				{
					cudaMemcpyAsync(d_HSI0,host_HSI+i*Dim,sizeof(double)*num_last0*Dim,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_C_cur0,host_C_cur,sizeof(double)*num_tol*Dim*num_last0,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_inv_CTC0,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_last0,cudaMemcpyHostToDevice,stream0);

					cudaMemcpyAsync(d_HSI1,host_HSI+(i+num_last0)*Dim,sizeof(double)*num_last1*Dim,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_C_cur1,host_C_cur,sizeof(double)*num_tol*Dim*num_last1,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_inv_CTC1,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_last1,cudaMemcpyHostToDevice,stream1);

					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock,0,stream0>>>(d_C_cur0,d_inv_CTC0,d_HSI0,Dim,num_tol,num_last0,d_x0,d_res0);
					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_C_cur1,d_inv_CTC1,d_HSI1,Dim,num_tol,num_last1,d_x1,d_res1);
				
					cudaMemcpyAsync(host_x+i*num_tol,d_x0,sizeof(double)*num_tol*num_last0,cudaMemcpyDeviceToHost,stream0);
					cudaMemcpyAsync(host_res+i,d_res0,sizeof(double)*num_last1,cudaMemcpyDeviceToHost,stream0);

					cudaMemsetAsync(d_x0,0,num_tol*num_last0*sizeof(double),stream0);
					cudaMemsetAsync(d_res0,0,num_last0*sizeof(double),stream0);
				 
					cudaMemcpyAsync(host_x+(i+num_last0)*num_tol,d_x1,sizeof(double)*num_tol*num_last1,cudaMemcpyDeviceToHost,stream1);
					cudaMemcpyAsync(host_res+(i+num_last0),d_res1,sizeof(double)*num_last1,cudaMemcpyDeviceToHost,stream1);

					cudaMemsetAsync(d_x1,0,num_tol*num_last1*sizeof(double),stream1);
					cudaMemsetAsync(d_res1,0,num_last1*sizeof(double),stream1);
				}
			}

			cudaStreamSynchronize(stream0);
			cudaStreamSynchronize(stream1);

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime,start,stop);
			printf("GPU-Time-comsuming: %lf sec\n",elapsedTime/1000.f);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			cudaFreeHost(host_inv_UTU);
			cudaFreeHost(host_C_cur);
			cudaFreeHost(host_HSI);



			printf("GPU: x_10: %lf %lf %lf %lf %lf\n",host_x[show],host_x[show+1],host_x[show+2],host_x[show+3],host_x[show+4]);
	
			cudaFreeHost(host_res);
			cudaFree(d_HSI0);cudaFree(d_HSI1);
			cudaFree(C_ori);
			cudaFree(d_C_cur0);cudaFree(d_C_cur1);
			cudaFree(d_inv_CTC0);cudaFree(d_inv_CTC1);
			cudaFree(d_x0);cudaFree(d_x1);
			cudaFree(d_res0);cudaFree(d_res1);
			cudaStreamDestroy(stream0);
			cudaStreamDestroy(stream1);
			//printf("Speedup: %lf times\n",duration*1000.0/elaspedTime);
			system("pause");
			MATFile *write_file=matOpen(argv[3],"w");
			size_t dims[3]={H,W,num_tol};
			mxArray *mat_array=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
			double * dst = (double *)(mxGetPr(mat_array));    
			double * d = dst;   
			for (int i = 0; i<num_tol;i++)
				for(int j=0;j<W;j++)
					for(int k=0;k<H;k++)
						(*d++)=host_x[j*H*num_tol+k*num_tol+i];
			cudaFreeHost(host_x);
			matPutVariable(write_file, "abundance", mat_array);
	
			matClose(write_file);
		}
		if(flag2&&!flag1)
		{
		//CPU-beginning
		clock_t start_time,end_time;
		double* res_cpu=new double[H*W];
		double* x_cpu=new double[H*W*num_tol];
		start_time=clock();
		for(int i=0;i<H*W;i++)
		{
			res_cpu[i]=fnnls(Endmember,HSI_re+i*Dim,Dim,num_tol,x_cpu+i*num_tol,1e-6);
		}
		end_time=clock();
		double duration = (double)(end_time-start_time) / CLOCKS_PER_SEC;
		printf( "CPU-Time-comsuming: %lf seconds\n", duration );
		printf("CPU: x_10: %lf %lf %lf %lf %lf\n",x_cpu[show],x_cpu[show+1],x_cpu[show+2],x_cpu[show+3],x_cpu[show+4]);
		delete[]res_cpu;res_cpu=NULL;
		delete[]x_cpu;x_cpu=NULL;
		double *d_HSI;
		cudaMalloc((void**)&d_HSI,W*H*Dim*sizeof(double));
		cudaMemcpy(d_HSI,HSI_re,Dim*H*W*sizeof(double),cudaMemcpyHostToDevice);
		//double *d_endmember;
		//cudaMalloc((void**)&d_endmember,num_tol*Dim*sizeof(double));
		//cudaMemcpy(d_endmember,Endmember,num_tol*Dim*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(C_ori,Endmember,sizeof(double)*num_tol*Dim);

		int num_threads=2000;
		double *d_C_cur;
		cudaMalloc((void**)&d_C_cur,(num_tol)*Dim*num_threads*sizeof(double));
		double* C_cur=new double[num_tol*Dim*num_threads];
		for(int i=0;i<num_threads;i++)
			for(int j=0;j<num_tol*Dim;j++)
				C_cur[i*num_tol*Dim+j]=Endmember[j];
		cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
		

		double *d_inv_CTC;
		cudaMalloc((void**)&d_inv_CTC,(num_tol)*(num_tol)*num_threads*sizeof(double));		
		double* inv_UTU=new double[num_tol*num_tol];
		double* UTU=new double[num_tol*num_tol];
		for(int i=0;i<num_tol;i++)
		{
			for(int j=0;j<=i;j++)
			{
				UTU[i*num_tol+j]=0;
				for(int k=0;k<Dim;k++)
					UTU[i*num_tol+j]+=Endmember[i*Dim+k]*Endmember[j*Dim+k];
				UTU[j*num_tol+i]=UTU[i*num_tol+j];
			}
		}

		inv_A(UTU,num_tol,inv_UTU);
		//cula_inv_A(UTU,num_tol,inv_UTU);//inv_A(UTU,num_tol,inv_UTU);//
 		double*_inv_UTU=new double[num_tol*num_tol*num_threads];
		for(int i=0;i<num_threads;i++)
			for(int j=0;j<num_tol*num_tol;j++)
				_inv_UTU[i*num_tol*num_tol+j]=inv_UTU[j];
		cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
		mxFree(Endmember);
		
		delete[]inv_UTU;inv_UTU=NULL;
		delete[]UTU;UTU=NULL;

		double *d_x;
		cudaMalloc((void**)&d_x,num_threads*num_tol*sizeof(double));
		double *d_res;
		cudaMalloc((void**)&d_res,num_threads*sizeof(double));

		double* x=new double[H*W*num_tol];
		double* res=new double[H*W];

		int iter=H*W/num_threads;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		if(H*W<=num_threads)
		{
			e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC,d_HSI,Dim,num_tol,H*W,d_x,d_res);
			cudaMemcpy(x,d_x,sizeof(double)*H*W*num_tol,cudaMemcpyDeviceToHost);
			cudaMemcpy(res,d_res,sizeof(double)*H*W,cudaMemcpyDeviceToHost);
		}
		else if(H*W%num_threads!=0)
		{
			for(int i=0;i<iter+1;i++)
			{

				if(i==iter)
				{
					int num_least=H*W%num_threads;
					cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
					cudaMemset(d_res,0,num_threads*sizeof(double));
					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_least,d_x,d_res);
					cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_least*num_tol,cudaMemcpyDeviceToHost);
					cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_least,cudaMemcpyDeviceToHost);
				}
				else
				{
					cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
					cudaMemset(d_res,0,num_threads*sizeof(double));
					e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res);
					cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
					cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
				}
			}
		}
		else
		{
			for(int i=0;i<iter;i++)
			{		
				cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
				cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
				cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
				cudaMemset(d_res,0,num_threads*sizeof(double));
				e_fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res);
				cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
				cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
			}
		}
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		float elaspedTime;
		cudaEventElapsedTime(&elaspedTime,start,stop);
		printf("GPU-Time-comsuming: %lf sec\n",elaspedTime/1000.f);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		delete[]_inv_UTU;_inv_UTU=NULL;
		delete[]C_cur;C_cur=NULL;

		printf("GPU: x_10: %lf %lf %lf %lf %lf\n",x[show],x[show+1],x[show+2],x[show+3],x[show+4]);
	
		delete[]res;res=NULL;
		cudaFree(d_HSI);
		cudaFree(C_ori);
		cudaFree(d_C_cur);
		cudaFree(d_inv_CTC);
		cudaFree(d_x);
		cudaFree(d_res);
		printf("Speedup: %lf times\n",duration*1000.0/elaspedTime);
		system("pause");
		MATFile *write_file=matOpen(argv[3],"w");
		size_t dims[3]={H,W,num_tol};
		mxArray *mat_array=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
		double * dst = (double *)(mxGetPr(mat_array));    
		double * d = dst;   
		for (int i = 0; i<num_tol;i++)
			for(int j=0;j<W;j++)
				for(int k=0;k<H;k++)
					(*d++)=x[j*H*num_tol+k*num_tol+i];
		delete[]x;x=NULL;
		matPutVariable(write_file, "abundance", mat_array);
	
		matClose(write_file);
		}
		//GPU-beginning-1
		else if(!flag2&&!flag1)
		{		
		double *d_HSI;
		cudaMalloc((void**)&d_HSI,W*H*Dim*sizeof(double));
		cudaMemcpy(d_HSI,HSI_re,Dim*H*W*sizeof(double),cudaMemcpyHostToDevice);
		//double *d_endmember;
		//cudaMalloc((void**)&d_endmember,num_tol*Dim*sizeof(double));
		//cudaMemcpy(d_endmember,Endmember,num_tol*Dim*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(C_ori,Endmember,sizeof(double)*num_tol*Dim);
		mxFree(Endmember);

		int num_threads=2000;
		double *d_C_cur;
		cudaMalloc((void**)&d_C_cur,num_tol*Dim*num_threads*sizeof(double));
		double *d_inv_CTC;
		cudaMalloc((void**)&d_inv_CTC,num_tol*num_tol*num_threads*sizeof(double));
		double *d_x;
		cudaMalloc((void**)&d_x,num_threads*num_tol*sizeof(double));
		double *d_res;
		cudaMalloc((void**)&d_res,num_threads*sizeof(double));

		double* x=new double[H*W*num_tol];
		double* res=new double[H*W];

		int iter=H*W/num_threads;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
		if(H*W<=num_threads)
		{
			fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC,d_HSI,Dim,num_tol,H*W,d_x,d_res,1.0e-6);
			cudaMemcpy(x,d_x,sizeof(double)*H*W*num_tol,cudaMemcpyDeviceToHost);
			cudaMemcpy(res,d_res,sizeof(double)*H*W,cudaMemcpyDeviceToHost);
		}
		else if(H*W%num_threads!=0)
		{
			for(int i=0;i<iter+1;i++)
			{

				if(i==iter)
				{
					int num_least=H*W%num_threads;
					cudaMemset(d_C_cur,0,num_tol*Dim*num_threads*sizeof(double));
					cudaMemset(d_inv_CTC,0,num_tol*num_tol*num_threads*sizeof(double));
					cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
					cudaMemset(d_res,0,num_threads*sizeof(double));
					fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_least,d_x,d_res,1.0e-6);
					cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_least*num_tol,cudaMemcpyDeviceToHost);
					cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_least,cudaMemcpyDeviceToHost);
				}
				else
				{
					cudaMemset(d_C_cur,0,num_tol*Dim*num_threads*sizeof(double));
					cudaMemset(d_inv_CTC,0,num_tol*num_tol*num_threads*sizeof(double));
					cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
					cudaMemset(d_res,0,num_threads*sizeof(double));
					fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res,1.0e-6);
					cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
					cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
				}
			}
		}
		else
		{
			for(int i=0;i<iter;i++)
			{		
				cudaMemset(d_C_cur,0,num_tol*Dim*num_threads*sizeof(double));
				cudaMemset(d_inv_CTC,0,num_tol*num_tol*num_threads*sizeof(double));
				cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
				cudaMemset(d_res,0,num_threads*sizeof(double));
				fnnls_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res,1.0e-6);
				cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
				cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
			}
		}
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		float elaspedTime;
		cudaEventElapsedTime(&elaspedTime,start,stop);
		printf("GPU-Time-comsuming: %lf sec\n",elaspedTime/1000.f);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("GPU: x_10: %lf %lf %lf %lf %lf\n",x[show],x[show+1],x[show+2],x[show+3],x[show+4]);	
		delete[]res;res=NULL;
		cudaFree(d_HSI);
		cudaFree(C_ori);
		cudaFree(d_C_cur);
		cudaFree(d_inv_CTC);
		cudaFree(d_x);
		cudaFree(d_res);
		//printf("Speedup: %lf times\n",duration*1000.0/elaspedTime);
		system("pause");
		MATFile *write_file=matOpen(argv[3],"w");
		size_t dims[3]={H,W,num_tol};
		mxArray *mat_array=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
		double * dst = (double *)(mxGetPr(mat_array));    
		double * d = dst;   
		for (int i = 0; i<num_tol;i++)
			for(int j=0;j<W;j++)
				for(int k=0;k<H;k++)
					(*d++)=x[j*H*num_tol+k*num_tol+i];
		delete[]x;x=NULL;
		matPutVariable(write_file, "abundance", mat_array);	
		matClose(write_file);
		}
	}
	else
	{
		if(flag2)
		{
			//CPU-beginning
			clock_t start_time,end_time;
			double* res_cpu=new double[H*W];
			double* x_cpu=new double[H*W*num_tol];

			start_time=clock();
			for(int i=0;i<H*W;i++)
			{
				res_cpu[i]=hyperFcls(Endmember,num_tol,Dim,HSI_re+i*Dim,x_cpu+i*num_tol);
			}
			end_time=clock();
			double duration = (double)(end_time-start_time) / CLOCKS_PER_SEC;
			printf( "CPU-Time-comsuming: %lf seconds\n", duration );
			printf("CPU: x_10: %lf %lf %lf %lf %lf\n",x_cpu[show],x_cpu[show+1],x_cpu[show+2],x_cpu[show+3],x_cpu[show+4]);
			delete[]res_cpu;res_cpu=NULL;
			delete[]x_cpu;x_cpu=NULL;
			cudaDeviceProp prop;
			int whichDevice;
			cudaGetDevice(&whichDevice);
			cudaGetDeviceProperties(&prop,whichDevice);
			if(!prop.deviceOverlap)
			{
				printf("Device will not handle overlaps, so no \"speed up from stream\"\n.");
				return 0;
			}
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaStream_t stream0,stream1;
			cudaStreamCreate(&stream0);
			cudaStreamCreate(&stream1);

			int num_threads=2048;
			int iter=H*W/(2*num_threads);
			int num_last=(H*W)%(2*num_threads);
			if(num_last==0)
			{
				iter--;
				num_last=2*num_threads;
			}
			int iteration=0;
			int num_last0,num_last1;
			num_last0=num_last1=num_last/2;
			if(num_last%2==1)
			{
				num_last1=num_last0+1;
			}


			double *host_HSI;
			cudaHostAlloc((void**)&host_HSI,W*H*Dim*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<W*H*Dim;i++)
				host_HSI[i]=HSI_re[i];
			double *d_HSI0,*d_HSI1;
			cudaMalloc((void**)&d_HSI0,num_threads*Dim*sizeof(double));
			cudaMalloc((void**)&d_HSI1,num_threads*Dim*sizeof(double));
			cudaMemcpyToSymbol(C_ori,Endmember,sizeof(double)*num_tol*Dim);

			double *d_C_cur0,*d_C_cur1;
			cudaMalloc((void**)&d_C_cur0,(num_tol)*Dim*num_threads*sizeof(double));
			cudaMalloc((void**)&d_C_cur1,(num_tol)*Dim*num_threads*sizeof(double));
			double* host_C_cur;
			cudaHostAlloc((void**)&host_C_cur,num_threads*Dim*num_tol*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*Dim;j++)
					host_C_cur[i*num_tol*Dim+j]=Endmember[j];

			double *d_inv_CTC0,*d_inv_CTC1;
			cudaMalloc((void**)&d_inv_CTC0,(num_tol)*(num_tol)*num_threads*sizeof(double));
			cudaMalloc((void**)&d_inv_CTC1,(num_tol)*(num_tol)*num_threads*sizeof(double));

			double* inv_UTU=new double[num_tol*num_tol];
			double* UTU=new double[num_tol*num_tol];
			for(int i=0;i<num_tol;i++)
			{
				for(int j=0;j<=i;j++)
				{
					UTU[i*num_tol+j]=0;
					for(int k=0;k<Dim;k++)
						UTU[i*num_tol+j]+=Endmember[i*Dim+k]*Endmember[j*Dim+k];
					UTU[j*num_tol+i]=UTU[i*num_tol+j];
				}
			}	
			//cula_inv_A(UTU,num_tol,inv_UTU);//inv_A(UTU,num_tol,inv_UTU);//
			inv_A(UTU,num_tol,inv_UTU);
			double *host_inv_UTU;
			cudaHostAlloc((void**)&host_inv_UTU,num_threads*num_tol*num_tol*sizeof(double),cudaHostAllocDefault);
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*num_tol;j++)
					host_inv_UTU[i*num_tol*num_tol+j]=inv_UTU[j];
			mxFree(Endmember);	
			delete[]inv_UTU;inv_UTU=NULL;
			delete[]UTU;UTU=NULL;

			double *d_x0,*d_x1;
			cudaMalloc((void**)&d_x0,num_threads*num_tol*sizeof(double));
			cudaMalloc((void**)&d_x1,num_threads*num_tol*sizeof(double));
			double *d_res0,*d_res1;
			cudaMalloc((void**)&d_res0,num_threads*sizeof(double));
			cudaMalloc((void**)&d_res1,num_threads*sizeof(double));
			double* host_x;cudaHostAlloc((void**)&host_x,W*H*num_tol*sizeof(double),cudaHostAllocDefault);
			double* host_res;cudaHostAlloc((void**)&host_res,W*H*sizeof(double),cudaHostAllocDefault);

			cudaEventRecord(start,0);
			
			for(int i=0;i<H*W;i+=(2*num_threads))
			{
				if(iteration!=iter)
				{
					cudaMemcpyAsync(d_HSI0,host_HSI+i*Dim,sizeof(double)*num_threads*Dim,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_C_cur0,host_C_cur,sizeof(double)*num_tol*Dim*num_threads,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_inv_CTC0,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_threads,cudaMemcpyHostToDevice,stream0);

					cudaMemcpyAsync(d_HSI1,host_HSI+(i+num_threads)*Dim,sizeof(double)*num_threads*Dim,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_C_cur1,host_C_cur,sizeof(double)*num_tol*Dim*num_threads,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_inv_CTC1,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_threads,cudaMemcpyHostToDevice,stream1);

					FCLS_GPU<<<blocksPerGrid,threadsPerBlock,0,stream0>>>(d_C_cur0,d_inv_CTC0,d_HSI0,Dim,num_tol,num_threads,d_x0,d_res0);
					FCLS_GPU<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_C_cur1,d_inv_CTC1,d_HSI1,Dim,num_tol,num_threads,d_x1,d_res1);
				
					cudaMemcpyAsync(host_x+i*num_tol,d_x0,sizeof(double)*num_tol*num_threads,cudaMemcpyDeviceToHost,stream0);
					cudaMemcpyAsync(host_res+i,d_res0,sizeof(double)*num_threads,cudaMemcpyDeviceToHost,stream0);

					//cudaMemsetAsync(d_x0,0,num_tol*num_threads*sizeof(double),stream0);
					//cudaMemsetAsync(d_res0,0,num_threads*sizeof(double),stream0);
				 
					cudaMemcpyAsync(host_x+(i+num_threads)*num_tol,d_x1,sizeof(double)*num_tol*num_threads,cudaMemcpyDeviceToHost,stream1);
					cudaMemcpyAsync(host_res+(i+num_threads),d_res1,sizeof(double)*num_threads,cudaMemcpyDeviceToHost,stream1);

					//cudaMemsetAsync(d_x1,0,num_tol*num_threads*sizeof(double),stream1);
					//cudaMemsetAsync(d_res1,0,num_threads*sizeof(double),stream1);
					iteration++;
				}
				else
				{
					cudaMemcpyAsync(d_HSI0,host_HSI+i*Dim,sizeof(double)*num_last0*Dim,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_C_cur0,host_C_cur,sizeof(double)*num_tol*Dim*num_last0,cudaMemcpyHostToDevice,stream0);
					cudaMemcpyAsync(d_inv_CTC0,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_last0,cudaMemcpyHostToDevice,stream0);

					cudaMemcpyAsync(d_HSI1,host_HSI+(i+num_last0)*Dim,sizeof(double)*num_last1*Dim,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_C_cur1,host_C_cur,sizeof(double)*num_tol*Dim*num_last1,cudaMemcpyHostToDevice,stream1);
					cudaMemcpyAsync(d_inv_CTC1,host_inv_UTU,sizeof(double)*num_tol*num_tol*num_last1,cudaMemcpyHostToDevice,stream1);

					FCLS_GPU<<<blocksPerGrid,threadsPerBlock,0,stream0>>>(d_C_cur0,d_inv_CTC0,d_HSI0,Dim,num_tol,num_last0,d_x0,d_res0);
					FCLS_GPU<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(d_C_cur1,d_inv_CTC1,d_HSI1,Dim,num_tol,num_last1,d_x1,d_res1);
				
					cudaMemcpyAsync(host_x+i*num_tol,d_x0,sizeof(double)*num_tol*num_last0,cudaMemcpyDeviceToHost,stream0);
					cudaMemcpyAsync(host_res+i,d_res0,sizeof(double)*num_last1,cudaMemcpyDeviceToHost,stream0);

					//cudaMemsetAsync(d_x0,0,num_tol*num_last0*sizeof(double),stream0);
					//cudaMemsetAsync(d_res0,0,num_last0*sizeof(double),stream0);
				 
					cudaMemcpyAsync(host_x+(i+num_last0)*num_tol,d_x1,sizeof(double)*num_tol*num_last1,cudaMemcpyDeviceToHost,stream1);
					cudaMemcpyAsync(host_res+(i+num_last0),d_res1,sizeof(double)*num_last1,cudaMemcpyDeviceToHost,stream1);

					//cudaMemsetAsync(d_x1,0,num_tol*num_last1*sizeof(double),stream1);
					//cudaMemsetAsync(d_res1,0,num_last1*sizeof(double),stream1);
				}
			}

			cudaStreamSynchronize(stream0);
			cudaStreamSynchronize(stream1);

			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			float elapsedTime;
			cudaEventElapsedTime(&elapsedTime,start,stop);
			printf("GPU-Time-comsuming: %lf sec\n",elapsedTime/1000.f);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);

			cudaFreeHost(host_inv_UTU);
			cudaFreeHost(host_C_cur);
			cudaFreeHost(host_HSI);



			printf("GPU: x_10: %lf %lf %lf %lf %lf\n",host_x[show],host_x[show+1],host_x[show+2],host_x[show+3],host_x[show+4]);
	
			cudaFreeHost(host_res);
			cudaFree(d_HSI0);cudaFree(d_HSI1);
			cudaFree(C_ori);
			cudaFree(d_C_cur0);cudaFree(d_C_cur1);
			cudaFree(d_inv_CTC0);cudaFree(d_inv_CTC1);
			cudaFree(d_x0);cudaFree(d_x1);
			cudaFree(d_res0);cudaFree(d_res1);
			cudaStreamDestroy(stream0);
			cudaStreamDestroy(stream1);
			//printf("Speedup: %lf times\n",duration*1000.0/elaspedTime);
			system("pause");
			MATFile *write_file=matOpen(argv[3],"w");
			size_t dims[3]={H,W,num_tol};
			mxArray *mat_array=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
			double * dst = (double *)(mxGetPr(mat_array));    
			double * d = dst;   
			for (int i = 0; i<num_tol;i++)
				for(int j=0;j<W;j++)
					for(int k=0;k<H;k++)
						(*d++)=host_x[j*H*num_tol+k*num_tol+i];
			cudaFreeHost(host_x);
			matPutVariable(write_file, "abundance", mat_array);
	
			matClose(write_file);
		}
		else
		{
			//CPU-beginning
			/*clock_t start_time,end_time;
			double* res_cpu=new double[H*W];
			double* x_cpu=new double[H*W*num_tol];

			start_time=clock();
			for(int i=0;i<H*W;i++)
			{
				res_cpu[i]=hyperFcls(Endmember,num_tol,Dim,HSI_re+i*Dim,x_cpu+i*num_tol);
			}
			end_time=clock();
			double duration = (double)(end_time-start_time) / CLOCKS_PER_SEC;
			printf( "CPU-Time-comsuming: %lf seconds\n", duration );
			printf("CPU: x_10: %lf %lf %lf %lf %lf\n",x_cpu[show],x_cpu[show+1],x_cpu[show+2],x_cpu[show+3],x_cpu[show+4]);
			delete[]res_cpu;res_cpu=NULL;
			delete[]x_cpu;x_cpu=NULL;*/
			//GPU-beginning
			double *d_HSI;
			cudaMalloc((void**)&d_HSI,W*H*Dim*sizeof(double));
			cudaMemcpy(d_HSI,HSI_re,Dim*H*W*sizeof(double),cudaMemcpyHostToDevice);
			//double *d_endmember;
			//cudaMalloc((void**)&d_endmember,num_tol*Dim*sizeof(double));
			//cudaMemcpy(d_endmember,Endmember,num_tol*Dim*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(C_ori,Endmember,sizeof(double)*num_tol*Dim);

			int num_threads=3000;
			double *d_C_cur;
			cudaMalloc((void**)&d_C_cur,(num_tol)*Dim*num_threads*sizeof(double));
			double* C_cur=new double[num_tol*Dim*num_threads];
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*Dim;j++)
					C_cur[i*num_tol*Dim+j]=Endmember[j];
			cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
		

			double *d_inv_CTC;
			cudaMalloc((void**)&d_inv_CTC,(num_tol)*(num_tol)*num_threads*sizeof(double));		
			double* inv_UTU=new double[num_tol*num_tol];
			double* UTU=new double[num_tol*num_tol];
			for(int i=0;i<num_tol;i++)
			{
				for(int j=0;j<=i;j++)
				{
					UTU[i*num_tol+j]=0;
					for(int k=0;k<Dim;k++)
						UTU[i*num_tol+j]+=Endmember[i*Dim+k]*Endmember[j*Dim+k];
					UTU[j*num_tol+i]=UTU[i*num_tol+j];
				}
			}
		
			//cula_inv_A(UTU,num_tol,inv_UTU);//inv_A(UTU,num_tol,inv_UTU);//
			inv_A(UTU,num_tol,inv_UTU);
			double*_inv_UTU=new double[num_tol*num_tol*num_threads];
			for(int i=0;i<num_threads;i++)
				for(int j=0;j<num_tol*num_tol;j++)
					_inv_UTU[i*num_tol*num_tol+j]=inv_UTU[j];
			cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
			mxFree(Endmember);
		
			delete[]inv_UTU;inv_UTU=NULL;
			delete[]UTU;UTU=NULL;

			double *d_x;
			cudaMalloc((void**)&d_x,num_threads*num_tol*sizeof(double));
			double *d_res;
			cudaMalloc((void**)&d_res,num_threads*sizeof(double));

			double* x=new double[H*W*num_tol];
			double* res=new double[H*W];

			int iter=H*W/num_threads;
			cudaEvent_t start,stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			cudaEventRecord(start,0);
			if(H*W<=num_threads)
			{
				FCLS_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC,d_HSI,Dim,num_tol,H*W,d_x,d_res);
				cudaMemcpy(x,d_x,sizeof(double)*H*W*num_tol,cudaMemcpyDeviceToHost);
				cudaMemcpy(res,d_res,sizeof(double)*H*W,cudaMemcpyDeviceToHost);
			}
			else if(H*W%num_threads!=0)
			{
				for(int i=0;i<iter+1;i++)
				{

					if(i==iter)
					{
						int num_least=H*W%num_threads;
						cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
						cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
						cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
						cudaMemset(d_res,0,num_threads*sizeof(double));
						FCLS_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_least,d_x,d_res);
						cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_least*num_tol,cudaMemcpyDeviceToHost);
						cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_least,cudaMemcpyDeviceToHost);
					}
					else
					{
						cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
						cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
						cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
						cudaMemset(d_res,0,num_threads*sizeof(double));
						FCLS_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res);
						cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
						cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
					}
				}
			}
			else
			{
				for(int i=0;i<iter;i++)
				{		
					cudaMemcpy(d_inv_CTC,_inv_UTU,num_tol*num_tol*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemcpy(d_C_cur,C_cur,num_tol*Dim*num_threads*sizeof(double),cudaMemcpyHostToDevice);
					cudaMemset(d_x,0,num_tol*num_threads*sizeof(double));
					cudaMemset(d_res,0,num_threads*sizeof(double));
					FCLS_GPU<<<blocksPerGrid,threadsPerBlock>>>(d_C_cur,d_inv_CTC, d_HSI+i*num_threads*Dim,Dim,num_tol,num_threads,d_x,d_res);
					cudaMemcpy(x+i*num_threads*num_tol,d_x,sizeof(double)*num_threads*num_tol,cudaMemcpyDeviceToHost);
					cudaMemcpy(res+i*num_threads,d_res,sizeof(double)*num_threads,cudaMemcpyDeviceToHost);
				}
			}
			cudaEventRecord(stop,0);
			cudaEventSynchronize(stop);
			float elaspedTime;
			cudaEventElapsedTime(&elaspedTime,start,stop);
			printf("GPU-Time-comsuming: %lf sec\n",elaspedTime/1000.f);
			cudaEventDestroy(start);
			cudaEventDestroy(stop);
			delete[]_inv_UTU;_inv_UTU=NULL;
			delete[]C_cur;C_cur=NULL;

			printf("GPU: x_10: %lf %lf %lf %lf %lf\n",x[show],x[show+1],x[show+2],x[show+3],x[show+4]);
	
			delete[]res;res=NULL;
			cudaFree(d_HSI);
			cudaFree(C_ori);
			cudaFree(d_C_cur);
			cudaFree(d_inv_CTC);
			cudaFree(d_x);
			cudaFree(d_res);
			//printf("Speedup: %lf times\n",duration*1000.0/elaspedTime);
			system("pause");
			MATFile *write_file=matOpen(argv[3],"w");
			size_t dims[3]={H,W,num_tol};
			mxArray *mat_array=mxCreateNumericArray(3, dims, mxDOUBLE_CLASS, mxREAL);
			double * dst = (double *)(mxGetPr(mat_array));    
			double * d = dst;   
			for (int i = 0; i<num_tol;i++)
				for(int j=0;j<W;j++)
					for(int k=0;k<H;k++)
						(*d++)=x[j*H*num_tol+k*num_tol+i];
			delete[]x;x=NULL;
			matPutVariable(write_file, "abundance", mat_array);
	
			matClose(write_file);
		}
	}
	delete[]HSI_re;HSI_re=NULL;
	return 0;
	}

