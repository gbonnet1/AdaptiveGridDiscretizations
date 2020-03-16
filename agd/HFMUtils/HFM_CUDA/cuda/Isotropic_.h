#pragma once 

const Int n_print = 100;
const Int n_print2=3;

#include "Constants.h"
#include "Grid.h"
#include "HFM.h"

extern "C" {

/**
	Integer parameters params_Int : 
	 - shape[ndim] : shape of domain
	 - shape_o[ndim] : outer shape
	Floating point parameters : 
	 - tol : convergence tolerance.
	 - step : avoid roundoff errors
	front_dist (uchar) : 
	 distance in blocks to the front. 
	 0 -> converged
	 1 -> converged,
	 ..-> not converged

*/
__global__ void IsotropicUpdate(Scalar * u, const Scalar * metric, const BoolPack * seeds, 
	const Scalar * paramsScalar,
	BoolAtom * updateNow_o, BoolAtom * updateNext_o){ // Used as simple booleans

//	__shared__ Int shape_o[ndim];
	__shared__ Int x_o[ndim];
	__shared__ Int n_o;
	__shared__ BoolAtom makeUpdate;
	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
		x_o[0]=blockIdx.x;     x_o[1]=blockIdx.y;     if(ndim==3) x_o[ndim-1]=blockIdx.z;
//		shape_o[0]=gridDim.x; shape_o[1]=gridDim.y; if(ndim==3) shape_o[ndim-1]=gridDim.z;
		n_o = Index(x_o,shape_o);
		makeUpdate = updateNow_o[n_o];
	}

	__syncthreads(); // __shared__ makeUpdate, ...
	if(!makeUpdate){return;}

	__shared__ Scalar tol;
	__shared__ Int shape[ndim];

	Int x_i[ndim], x[ndim];
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[ndim-1]=threadIdx.z;
	for(int k=0; k<ndim; ++k){
		x[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_i = Index(x_i,shape_i);
	const Int n = n_o*size_i + n_i;

	if(n_i==0){
		for(int k=0; k<ndim; ++k){
				shape[k] = shape_o[k]*shape_i[k];}
		updateNow_o[n_o] = 0;
		tol = paramsScalar[0];
		//	const Scalar step = params_Scalar[1]; // Avoid roundoff errors
	}



	const bool isSeed = GetBool(seeds,n);
	const Scalar cost = metric[n];
	const Scalar u_old = u[n];
	__shared__ Scalar u_i[size_i]; // Shared block values
	u_i[n_i] = u_old;

	__syncthreads(); // __shared__ shape. 
	// Get the neighbor values, or their indices if interior to the block
	Scalar v_o[ntot];
	Int    v_i[ntot];
	for(Int k=0,ks=0; k<nsym; ++k){
		for(Int s=0; s<2; ++s){
			Int * y = x; // Caution : aliasing
			Int * y_i = x_i;
			const Int eps=2*s-1;

			y[k]+=eps; y_i[k]+=eps;
			if(InRange(y_i,shape_i))  {v_i[ks] = Index(y_i,shape_i);}
			else {
				v_i[ks] = -1;
				if(InRange(y,shape)) {v_o[ks] = u[Index(y,shape_i,shape_o)];}
				else {v_o[ks] = infinity();}
			}
			y[k]-=eps; y_i[k]-=eps;
			++ks;
		}
	}
//	__syncthreads(); 

	// Compute and save the values
	HFMIter(!isSeed,n_i,cost,v_o,v_i,u_i);
	u[n] = u_i[n_i];
	
	// Find the smallest value which was changed.
	const Scalar u_diff = abs(u_old - u_i[n_i]);
	if( !(u_diff>tol) ){// Equivalent to u_diff<=tol, but Ignores NaNs 
		u_i[n_i]=infinity();}
	__syncthreads(); // Get all values before reduction

	Reduce_i( u_i[n_i] = min(u_i[n_i],u_i[m_i]); )
	__syncthreads();  // Make u_i[0] accessible to all 

	// Tag neighbor blocks for update
	if(u_i[0]!=infinity() && n_i<=2*ndim){ 
		Int k = n_i/2;
		const Int s = n_i%2;
		Int eps = 2*s-1;
		if(n_i==2*ndim){k=0; eps=0;}

		Int neigh_o[ndim];
		for(Int l=0; l<ndim; ++l) {neigh_o[l]=x_o[l];}
		neigh_o[k]+=eps;
		if(InRange(neigh_o,shape_o)) {updateNext_o[Index(neigh_o,shape_o)]=1;}
	}

	if(debug_print && n==0){
		printf("shape %i,%i",shape[0],shape[1]);

	}
/*
	if(debug_print && n==0){
		printf("tol %f\n",tol);
		printf("shape %i,%i\n",shape[0],shape[1]);
	}

	if(debug_print && n==0){
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
		printf("min_chg[0] %f\n",min_chg[0]);
	}
	if(debug_print && n_i==0){min_chg[blockIdx.x] = u_i[0];
		printf("Hello world %f %i\n", u_i[0],blockIdx.x);
		printf("min_chg[0] %f\n",min_chg[0]);
	}
	*/
}

} // Extern "C"