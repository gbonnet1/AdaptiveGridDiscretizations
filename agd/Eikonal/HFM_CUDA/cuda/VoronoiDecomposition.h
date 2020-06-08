// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file computes on the GPU and exports a Voronoi decomposition of a quadratic form.*/

/* // The following typedefs, or equivalent, must be defined in enclosing file
typedef int Int;
typedef float Scalar;
typedef char OffsetT;
*/

#if ndim_macro==2
#include "Geometry2.h"
#elif ndim_macro==3
#include "Geometry3.h"
#elif ndim_macro==4
#include "Geometry4.h"
#elif ndim_macro==5
#include "Geometry5.h"
#elif ndim_macro==6
#include "Geometry6.h"
#endif

__constant__ int size_tot;
const Int nsym = decompdim;

void scheme(const Scalar dual_metric[symdim], Scalar weights[nsym], OffsetT offsets[nsym][ndim]){
	// For some mysterious reason, decomp_m needs to be called from a __device__ function
	// otherwise cudaIllegalAddressError ?!? (Problem related with embedded lp solver ?!?)
	decomp_m(dual_metric,weights,offsets);
}

extern "C" {

__global__ void VoronoiDecomposition(const Scalar * __restrict__ m_t,
	Scalar * __restrict__ weights_t, OffsetT * offsets_t){
	const int n_i = threadIdx.x;
	const int n_o = blockIdx.x;
	const int n_t = n_o*blockDim.x + n_i;
	if(n_t>=size_tot) return;

	// Load the data
	Scalar m[symdim];
	Scalar weights[decompdim];
	OffsetT offsets[decompdim][ndim]; // geometry last
	for(Int i=0; i<symdim; ++i){m[i] = m_t[n_t+size_tot*i];}

	// Voronoi decomposition, and export
	scheme(m,weights,offsets); // Cannot call decomp_m directly
	
	for(Int i=0; i<decompdim; ++i){
		weights_t[n_t+i*size_tot]=weights[i];} // geometry first
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<decompdim; ++j){
			offsets_t[n_t+size_tot*(j+decompdim*i)]=offsets[j][i];}} // geometry first
}

} // Extern "C"