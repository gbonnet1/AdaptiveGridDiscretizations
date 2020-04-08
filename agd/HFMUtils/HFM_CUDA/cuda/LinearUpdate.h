// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a linear update operator, in a format somewhat similar to 
the HFM eikonal update operator. It is used to solve the linear systems arising in the 
automatic differentiation of the eikonal solver. These are triangular or almost triangular
systems solved using Gauss-Siedel iteration.
*/

#ifndef Scalar_macro
typedef float Scalar
#endif
Scalar infinity(){return 1./0.;}

#ifndef Int_macro
typedef int Int
#endif

#ifndef minchg_freeze_macro
#define minchg_freeze_macro 0
#endif
#if minchg_freeze_macro
#define MINCHG_FREEZE(...) __VA_ARGS__
#else
#define MINCHG_FREEZE(...) 
#endif

#ifndef pruning_macro
#define pruning_macro 0
#endif
#if pruning_macro
#define PRUNING(...) __VA_ARGS__
#else
#define PRUNING(...) 
#endif

#ifndef nrhs_macro
const Int nrhs=1;
#endif

#ifndef nindex_macro
const Int nindex = 4; // Number of entries per matrix line
#endif

#ifndef ndim_macro
const Int ndim=2;
#endif

#ifndef shape_i_macro
const Int shape_i[ndim] = {8,8};
const Int size_i = 64;
const Int log2_size_i = 7;
#endif

#ifndef niter_macro
const Int niter=16;
#endif

__constant__ Int shape_o; 
__constant__ Int size_o;
__constant__ Int size_tot;
__constant__ Scalar atol;
__constant__ Scalar rtol;

extern "C" {

void __global__ Update(
	Scalar * u_t, const Scalar * rhs_t, 
	const Scalar * diag_t, const Int * index_t, const Scalar * weight_t,
	MINCHG_FREEZE(const Scalar chg_t, const Scalar * minChgPrev_o, Scalar * minChgNext_o,)
	Int * updateList_o, PRUNING(BoolAtom * updatePrev_o,) BoolAtom * updateNext_o 
	){ 

	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if( Propagation::Abort(
		updateList_o,PRUNING(updatePrev_o,) 
		MINCHG_FREEZE(minChgPrev_o,updateNext_o)
		x_o,&n_o) ){return;} // Also sets x_o, n_o

	const Int n_i = threadIdx.x;
	const Int n_t = n_o*size_i + n_i;

	__shared__ Scalar u_i_[nrhs][size_i];
	Scalar rhs_[nrhs];
	Scalar u_old[nrhs];

	for(Int irhs=0; irhs<nrhs; ++irhs){
		u_old[irhs] = u_t[irhs*size_tot + n_t];
		u_i_[irhs][n_i] = u_old[irhs];
		rhs_[irhs] =    rhs_t[irhs*size_tot + n_t];
	}

	const Scalar diag = diag_t[n_t];
	Int    v_i[nindex]; // Index of neighbor, if in the block
	Scalar v_o_[nrhs][nindex]; // Value of neighbor, if outside the block
	Scalar weight[nindex];

	const Int v_i_inBlock = -1;
	const Int v_i_invalid = -2;

	for(Int k=0; k<nindex; ++k){
		weight[k] = weight_t[k*size_tot + n_t];
		if(weight[k]==0.) {v_i[k]=v_i_invalid; continue;}

		index = index_t[k*size_tot + n_t];
		if(ind/size_i == n_o){
			v_i[k] = ind%size_i;
		} else {
			v_i[k] = v_i_inBlock;
			for(Int irhs=0; irhs<nrhs; ++irhs){
				v_o_[irhs][k] = u_t[irhs*size_tot + index]}
		}

	}

	__syncthreads();

	// Gauss-Siedel iterations
	for(Int iter=0; iter<niter; ++iter){
		for(Int irhs=0; irhs<nrhs; ++irhs){
			Scalar * u_i = u_i[irhs];
			Scalar rhs = rhs[irhs]
			Scalar * v_o = v_o[irhs];

			// Accumulate the weighted neighbor values
			Scalar sum=0;
			for(Int k=0; k<nindex; ++k){
				const Scalar w_i = v_i[k];
				if(w_i==v_i_invalid) {continue;}
				const Scalar val = w_i==v_i_inBlock ? v_o[k] : u_i[w_i];
				sum += weight[k] * val;
			}

			// Normalize and store 
			u_i[n_i] = sum/diag;

		} // for irhs
		__syncthreads();
	} // for iter
	
	// Export and check for changes
	bool changed=false;
	for(Int irhs=0; irhs<nrhs; ++irhs){
		const Scalar val = u_i_[irhs][n_i];
		u_t[irhs*size_tot + n_t] = val;
		old = u_old[irhs]
		const Scalar tol = max(abs(val),abs(old))*rtol + atol;
		changed = changed || abs(val - old) > tol;
	}

	MINCHG_FREEZE(
	__shared__ Scalar chg_i[size_i];
	chg_i[n_i] = changed ? chg_t[n_t] : infinity();
	REDUCE_i(chg_i[n_i] = min(chg_i[n_i],chg_i[m_i]);)
	__syncthreads()
	)

	Propagation::Finalize(
		u_i, PRUNING(updateList_o,) 
		minChgPrev, MINCHG_FREEZE(minChgNext_o, 
		updatePrev_o,) updateNext_o,  
		x_o, n_o);

} // LinearUpdate


} // extern "C"