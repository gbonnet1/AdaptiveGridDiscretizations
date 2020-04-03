#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the inf-convolution of an array with a small constant array.
The implementation could be substantially optimized, by rearranging the data in blocks, etc
*/

#ifndef ndim_macro
const Int ndim=2;
#endif

#ifndef shape_c_macro
const Int shape_c[ndim] = {3,3};
const Int size_c = 3*3;
#endif

__constant__ Int shape_tot[ndim];
__constant__ Int size_tot; // product of shape_tot

#ifndef
typedef float T;
const T T_Max = 1./0.;
/* // Optionally use saturated arithmetic (for integral types)
#define saturation_macro
*/

__constant__ T kernel_c[size_c];

#include "Grid.h"

extern "C" {
__global__ void 
InfConvolution(const T * input, T * output){
	// Get the position where the work is to be done.
	const Int n_t = BlockIdx.x*BlockDim.x + ThreadIdx.x;
	if(n_t>=size_tot) {return;}
	Int x_t[ndim];
	Grid::Position(n_t,shape_tot,x_t);

	T result = T_Max;
	// Access neighbor values, and perform the inf convolution
	for(Int i_c=0; i_c<size_c; ++i_c){
		Int y_t[ndim];
		Grid::Position(i_c,shape_c,y_t);
		for(Int k=0; k<ndim; ++k){
			y_t[k] += x_t[k] - shape_c[k]/2;} // Centered kernel
		if(Grid::InRange_per(y_t,shape_tot)){
			const Int ny_t = Grid::Index_per(y_t);
			const T vy = input[ny_t];
			const T vc = kernel_c[i_c];

			#ifdef saturation_macro
			const T sum = (vy<=T_Max-vc) ? vy+vc : T_Max;
			#else
			const T sum = vy+vc;
			#endif

			result = min(result,sum);
		}
	}
	output[n_t] = result;
}

} // extern "C"

