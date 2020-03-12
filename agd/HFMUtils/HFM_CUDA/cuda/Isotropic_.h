const Int n_print = 100;
const Int n_print2=3;

#include "HFM.h"

extern "C" {

__global__ void IsotropicUpdate(Scalar * u, const Scalar * metric, const BoolPack * seeds, const Int * shape,
	const Int * _x_o, Scalar * min_chg, const Scalar tol){

	// Setup coordinate system
	Int x_i[ndim], x_o[ndim], x[ndim]; 
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[2]=threadIdx.z;
	const Int * __x_o = _x_o + ndim*blockIdx.x;
	for(int k=0; k<ndim; ++k){
		x_o[k] = __x_o[k];
		x[k] = x_o[k]*shape_i[k]+x_i[k];
	}

	GridType grid; // Share ?
	for(int k=0; k<ndim; ++k){
		grid.shape[k] = shape[k];
		grid.shape_o[k] = ceil_div(shape[k],shape_i[k]);
	}

	// Import local data
	const Int 
	n_i = grid.Index_i(x_i),
	n = grid.Index(x);

	const bool inRange = grid.InRange(x);
	const Scalar u_old = u[n];
	__shared__ Scalar u_i[size_i];
//	__shared__ Scalar u_new[size_i];
	u_i[n_i] = u_old;
//	u_new[n_i] = u_old;


	if(debug_print && n==n_print2){
		printf("n_print = %i\n",n_print);
		printf("Grid shape %i %i, %i %i\n", 
			grid.shape[0],grid.shape[1],
			grid.shape_o[0],grid.shape_o[1]);

		printf("x_i %i %i \n", x_i[0], x_i[1]);
		printf("x_o %i %i \n", x_o[0], x_o[1]);
		printf("x %i %i \n", x[0], x[1]);
		printf("Hello world");

	}

	const Scalar cost = metric[n];
	const bool active = (cost < infinity()) && (! GetBool(seeds,n));

	if(debug_print && n==n_print2){
		printf("inRange %i\n",inRange);
		printf("u_old %f\n",u_old);
		printf("cost %f\n",cost);
		printf("active %i\n",active);
		printf("u_i[n_i] %f\n",u_i[n_i]);

		printf("Seeds : %i %i %i %f %f\n",GetBool(seeds,0),GetBool(seeds,19),
			cost<infinity(),infinity(),not_a_number());
	}

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
			if(grid.InRange_i(y_i))  {v_i[ks] = grid.Index_i(y_i);}
			else {
				v_i[ks] = -1;
				if(grid.InRange(y)) {v_o[ks] = u[grid.Index(y)];}
				else {v_o[ks] = infinity();}
			}
			y[k]-=eps; y_i[k]-=eps;
		}
	}

	if(debug_print && n==n_print2){
		for(Int k=0; k<ndim; ++k){
			for(Int s=0; s<ndim; ++s){
				printf("(k%i,s%i) v_o %f, v_i %i \n", k,s, v_o[2*k+s],v_i[2*k+s]);
			}
		}

	}

	HFMIter(active,n_i,cost,v_o,v_i,u_i);
	/*
	// Make the updates
	for(int i=0; i<niter_i; ++i){
		if(active) {u_new[n_i] = HFMUpdate(n_i,cost,v_o,v_i,u_i);}
		__syncthreads();
		u_i[n_i]=u_new[n_i];
		__syncthreads();
	}*/
	u[n] = u_i[n_i];
	
	// Find the smallest value which was changed.
	Scalar u_diff = abs(u_old - u_i[n_i]);
	if( !(u_diff>tol) ){// Ignores NaNs (contrary to u_diff<=tol)
		u_i[n_i]=infinity();}

	if(debug_print && n==0){
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
	}

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
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
		printf("min_chg[0] %f\n",min_chg[0]);
	}
	if(debug_print && n_i==0){min_chg[blockIdx.x] = u_i[0];
		printf("Hello world %f %i\n", u_i[0],blockIdx.x);
		printf("min_chg[0] %f\n",min_chg[0]);
	}
}

} // Extern "C"