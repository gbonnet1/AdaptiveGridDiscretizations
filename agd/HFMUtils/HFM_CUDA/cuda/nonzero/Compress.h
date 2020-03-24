typedef int Int;

#ifndef shape_i_macro
#define shape_i_macro
const Int size_i = 1024; // BlockDim.x
const Int log2_size_i = 10; // Upper bound on log2(size_i)
#endif

#include "Accumulate.h"

//__constant__ Int size_tot;

extern "C" {

__global__ void Compress(const Int * index_in, Int * index, Int * nindex, Int size_tot){
	const Int size_o = gridDim.x;
	const Int n_o    = blockIdx.x;
	const Int n_i    = threadIdx.x;

	const Int n = n_i*size_o + n_o;
	const Int index_i = n>=size_tot ? -1 : index_in[n];
	const bool isActive = index_i!=-1;

	__shared__ Int active_acc_i[size_i];
	active_acc_i[n_i] = isActive;

	__syncthreads();
	Accumulate(active_acc_i,n_i,log2_size_i);

	if(n_i==size_i-1) {
		nindex[n_o] = active_acc_i[size_i-1];}

	if(isActive){
		const Int pos = (active_acc_i[n_i]-1)*size_o + n_o;
		index[pos] = index_i;}
} 

}