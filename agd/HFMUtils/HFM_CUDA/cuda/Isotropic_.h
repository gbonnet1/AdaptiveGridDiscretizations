#pragma once 

const Int n_print = 100;
const Int n_print2=3;

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
	const Int * paramsInt, const Scalar * paramsScalar,
	BoolPack * updateNow_o, BoolPack * updateNext_o){ // Used as simple booleans

	__shared__ Int shape_o[ndim];
	__shared__ Int x_o[ndim];
	__shared__ Int n_o;
	__shared__ BoolPack makeUpdate;
	if(n==0){
		x_o[0]=blockIdx.x;     x_o[1]=blockIdx.y;     if(ndim==3) x_o[2]=blockIdx.z;
		shape_o[0]=blockDim.x; shape_o[1]=blockDim.y; if(ndim==3) shape_o[2]=blockDim.z;
		n_o = Index(x_o,shape_o);
		makeUpdate = updateNow_o[n_o];
	}

	__syncthreads();
	if(!makeUpdate){return;}

	__shared__ Scalar tol;
	if(n==0){
		updateNow_o[n_o] = 0;
		tol = params_Scalar[0];
		//	const Scalar step = params_Scalar[1]; // Avoid roundoff errors
	}


	Int x_i[ndim], x[ndim];
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[2]=threadIdx.z;
	for(int k=0; k<ndim; ++k){x[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_i = Index(x_i,shape_i);
	const Int n = n_o*size_i + n_i;

	const bool isSeed = GetBool(seeds,n);
	const Scalar cost = metric[n];
	const Scalar u_old = u[n];
	__shared__ Scalar u_i[size_i]; // Shared block values
	u_i[n_i] = u_old;


/*
	// Setup coordinate system
	Int x_i[ndim], x_o[ndim], x[ndim]; 
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[2]=threadIdx.z;
	const Int * __x_o = _x_o + ndim*blockIdx.x;
	for(int k=0; k<ndim; ++k){
		x_o[k] = __x_o[k];
		x[k] = x_o[k]*shape_i[k]+x_i[k];
	}

	const Int n_i = Index(x_i,shape_i)
	if(n_i<ndim){shape[n_i] = params_Int[n_i];}
	if(ndim<=n_i && n_i<2*n_dim){shape_o[n_i-ndim] = params_Int[n_i];}
*/

	// Get the neighbor values, or their indices if interior to the block
	Scalar v_o[2*ndim];
	Int    v_i[2*ndim];
	for(Int k=0; k<ndim; ++k){
		for(Int s=0; s<2; ++s){
			Int * y = x; // Caution : aliasing
			Int * y_i = x_i;
			const Int eps=2*s-1;
			const Int ks = 2*k+s;

			y[k]+=eps; y_i[k]+=eps;
			if(InRange(y_i,shape_i))  {v_i[ks] = Index(y_i,shape_i);}
			else {
				v_i[ks] = -1;
				if(InRange(y,shape)) {v_o[ks] = u[Index(y,shape_i,shape_o)];}
				else {v_o[ks] = infinity();}
			}
			y[k]-=eps; y_i[k]-=eps;
		}
	}
	__syncthreads();

	// Compute and save the values
	HFMIter(active,n_i,cost,v_o,v_i,u_i);
	u[n] = u_i[n_i];
	
	// Find the smallest value which was changed.
	const Scalar u_diff = abs(u_old - u_i[n_i]);
	if( !(u_diff>tol) ){// Ignores NaNs (contrary to u_diff<=tol)
		u_i[n_i]=infinity();}
	__syncthreads();

/*
	if(debug_print && n_i==9){
		printf("n_i %i, u_old %f, u_new %f,\n",n_i,u_old,u_i[n_i]);
	}

*/
/*
	if(debug_print && n==0){
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
	}
*/


	Min0(n_i,u_i);
	/*
	Int shift=1;
	for(Int k=0; k<log2_size_i; ++k){
		Int old_shift=shift;
		shift=shift<<1;
		if( (n_i%shift)==0){
			Int m_i = n_i+old_shift;
			if(m_i<size_i){
				u_i[n_i] = min(u_i[n_i],u_i[m_i]);
			}
		}
		if(k<log2_size_i-1) {__syncthreads();}
	}*/
	if(n_i==0) {min_chg[blockIdx.x]=u_i[0];}

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
}

} // Extern "C"