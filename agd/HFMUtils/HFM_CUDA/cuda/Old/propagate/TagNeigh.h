#if dummy_minChg_macro
typedef unsigned char Scalar;
#else
typedef float Scalar;
__constant__ Scalar minChg_max;
#endif
typedef int Int;

/* // The followind constants must be defined.
const Int ndim = 2;
const Int shape[ndim] = {10,10};
*/

__constant__ Int index_size;

#define PERIODIC(...) 
const Int shape_i[3]={0,0,0}; const Int shape_o[3]={0,0,0}; const Int shape_tot[3]={0,0,0};
const Int size_i=0; typedef unsigned char BoolPack;
#include "../Grid.h"

extern "C" {
__global__ void TagNeigh(const Scalar * minChg, const Int * index, Int * tags){
	const Int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid>=index_size) return;
	
	const Int n = index[tid];
	#if dummy_minChg_macro
	const bool needsUpdate = minChg[n];
	#else
	const bool needsUpdate = minChg[n] < minChg_max;
	#endif
	
	if(!needsUpdate) return;

	tags[n]=n;

	// Diamond connectivity
	Int x[ndim];
	Grid::Position(n,shape,x);
	for(Int k=0; k<ndim; ++k){
		for(Int eps=-1; eps<=1; eps+=2){
			x[k]+=eps;
			if(Grid::InRange(x,shape)){
				tags[Grid::Index(x,shape)]=n;}
			x[k]-=eps;
		}
	} 
}
} // extern "C"