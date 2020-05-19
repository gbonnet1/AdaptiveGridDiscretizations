/** This file computes on the GPU and exports a Voronoi decomposition of a quadratic form.*/

#if ndim_macro==2
#include "Geometry2.h"
#elif ndim_macro==3
#include "Geometry3.h"
#elif ndim_macro==4
#include "Geometry4.h"
#elif ndim_macro==5
#include "Geometry5.h"
#endif


typedef char OffsetT;

extern "C" {

__global__ void decomposition(const Scalar * __restrict__ m,
	Scalar * __restrict__ weights, OffsetT * offsets){

	// Load the data

	// Call decomp_m

	// Export the weights and offsets

}


}