#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "NetworkSort.h"

/* This file is a GPU implementation of the simplify_ad method used to accumulate 
coefficients associated with identical indices in the ad.Sparse.spAD class.

Note : the method is embarassingly parallel, and threads do not interact.
*/


/* // The following, or variants, must be defined externally
typedef int IndexT; 
const Index IndexMax = ...; // Maximal value of an Int
typedef float Scalar;
const int bound_ad; // An upper bound on size_ad
#define atol_macro true
*/
__constant__ int size_ad;
__constant__ SizeT size_tot;
#if atol_macro
__constant__ Scalar atol = 0.;
#endif

extern "C" {

__global__ void simplify_ad(IndexT * __restrict__ index_t, Scalar * __restrict__ coef_t,
	int * __restrict__ new_size_ad_t){
	const SizeT n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t>=size_tot) return;

	// Load the data
	IndexT index[bound_ad];
	Scalar coef[bound_ad];
	for(Int i=0; i<size_ad; ++i){
		const Int i_t = n_t*size_ad+i;
		index[i] = index_t[i_t];
		coef[i]  = coef[i_t];
	}
	for(int i=size_ad; i<bound_ad; ++i){index[i] = IndexMax;}


	// Sort the indices
	int order[bound_ad];
	Network_sort::network_sort<bound_ad>(index,order);


	// Accumulate the coefficients associated with identical indices
	IndexT index_out[bound_ad];
	Scalar coef_out[bound_ad];

	int i_acc=0;
	index_out[0] = index[order[0]];
	coef_out[0] = coef[order[0]];

	for(int i=1; i<size_ad; ++i){
		const int j = order[i];
		ind = index[j];
		co  = coef[j];
		if(ind==index_out[i_acc]){
			coef_out[i_acc] += co;
		} else {
			i_acc+=1;
			index_out[i_acc] = ind;
			coef_out[i_acc]  = co;
	}

	int new_size_ad = i_acc+1;
	
	#if atol_macro
	// Discard coefficients which are below the specified threshold
	i_acc=0;
	for(int i=0; i<new_size_ad; ++i){
		if(abs(coef_out[i])<=atol){
			index_out[i_acc] = index_out[i];
			coef_out[i_acc]  = coef_out[i];
			i_acc += 1;
		}
	}
	new_size_ad = i_acc+1;
	#endif
	
	// Fill with dummy values the useless coefs
	const IndexT index_dummy = n_t;
	for(int i=new_size_ad; i<size_ad; ++i){
		index_out[i] = index_dummy;
		coef_out[i]  = 0.;
	}

	// Export the results
	for(int i=0; i<size_ad; ++i){
		const Int i_t = n_t*size_ad+i;
		index_t[i_t] = index_out[i];
		coef_t[i_t]  = coef_out[i];
	}
	new_size_ad_t[n_t] = new_size_ad;
}

} // extern "C"
