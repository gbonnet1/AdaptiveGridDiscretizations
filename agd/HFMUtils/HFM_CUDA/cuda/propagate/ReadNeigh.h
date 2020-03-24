typedef int Int;

/* // The followind constants must be defined.
const Int ndim = 2;
const Int shape[ndim] = {10,10};
*/

__constant__ Int index_size;

#include "../Grid.h"

extern "C" {
void TagNeigh(const Int * index, const Int * tags, Int * indexNext){
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

	const Int x = Grid::Position(n);
	for(Int k=0; k<ndim; ++k){
		for(Int eps=-1; eps<=1; eps+=2){
			x[k]+=eps;
			if(Grid::InRange(x)){
				const Int m = Grid::Index[x];
				if(tags[m]==n){
					neighIndex[iNeigh]=m;
					++iNeigh;
				}
			x[k]-=eps;
		}
	} 
}
} // extern "C"