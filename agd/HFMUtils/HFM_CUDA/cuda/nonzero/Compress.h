typedef int Int

__constant__ Int size_i; // BlockDim.x
__constant__ Int log2_size_i; // Upper bound on log2(size_i)
__constant__ Int size_tot;

#include "Accumulate.h"

extern "C" {

void Compress(const Int * index_in, Int * index, Int * nindex){
	const Int size_o = gridDim.x;
	const Int n_o    = blockIdx.x;
//	const Int size_i = blockDim.x;
	const Int n_i    = threadIdx.x;

	const Int n = n_i*size_o + n_o;
	const Int index = n>=size_tot ? -1 : index_in[n];
	const bool isActive = index!=-1;

	__shared__ Int active_acc_i[size_i];
	active_acc_i[n_i] = isActive;

	__syncthreads();
	Accumulate(active_acc_i,n_i,log2_size_i);

	if(n_i==size_i-1) {
		nindex[n_o] = active_acc_i[size_i-1];}

	if(isActive){
		const Int pos = active_acc_i[n_i]*size_o + n_o;
		index[pos] = n;}
} 

}