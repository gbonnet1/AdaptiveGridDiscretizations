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
__global__ void ReadNeigh(const Int * index, const Int * tags, Int * indexNext){
	const Int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if(tid>=index_size) return;
	
	// Diamond connectivity
	const Int nNeigh = 2*ndim+1;
	Int * neighIndex = &indexNext[nNeigh*tid];
	Int iNeigh=0;

	const Int n = index[tid];
	if(tags[n]==n){
		neighIndex[iNeigh]=n;
		++iNeigh;
	}

	Int x[ndim];
	Grid::Position(n,shape,x);
	for(Int k=0; k<ndim; ++k){
		for(Int eps=-1; eps<=1; eps+=2){
			x[k]+=eps;
			if(Grid::InRange(x,shape)){
				const Int m = Grid::Index(x,shape);
				if(tags[m]==n){
					neighIndex[iNeigh]=m;
					++iNeigh;
				}
			}
			x[k]-=eps;
		}
	} 
}
} // extern "C"