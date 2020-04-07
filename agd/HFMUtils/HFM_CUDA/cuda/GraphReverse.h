// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a weighted graph inversion method on the GPU.
It is used for the transposition of a triangular matrix, solved in a csr format.
*/

#ifndef Int_macro
typedef int Int;
#endif
#ifndef invalid_macro
const Int invalid = 2147483647;
#endif

#ifndef irevT_macro
typedef char irevT;
#endif

#ifndef weightT_macro
typedef float weightT;
#endif

const irevT irev_done = -1;

__constant__ Int size_tot;
__constant__ Int nfwd;
__constant__ Int nrev; // must be less than irevT_Max

extern "C" {

__global__ void GraphReverse(
	const Int * fwd_t, Int * rev_t, irevT * irev_t
	const weightT * fwd_weight_t, weightT * rev_weight_t){

	const Int n_t = blockIdx.x*blockDim.x + threadIdx.x;
	if(n_t >= size_tot){return;}

	for(Int ifwd=0; ifwd<nfwd; ++ifwd){
		const Int nfwd_t = n_t + ifwd*size_tot;

		// Check if there is anything to do
		const irevT irev = irev_t[nfwd_t];
		if(irev==irev_done) continue;
		if(irev==nrev) continue; // Arrays need to be resized externally

		const Int fwd = fwd_t[nfwd_t];
		if(fwd==invalid) continue;

		// Check if the edge is already correctly inverted		
		const Int nrev_t = n_t + irev*size_tot;
		const Int rev = rev_t[nrev_t];

		if(rev==fwd){ // Finish the work by copying the edge weight
			rev_weight_t[nrev_t] = fwd_weight_t[nfwd_t];
			irev_t[nfwd_t] = irev_done;
			continue;
		}

		// Find some place to copy
		while(irev<nrev){
			if(rev==invalid){ // Empty place
				rev_t[nrev_t] = fwd;
				irev_t[nfwd_t] = rev;
				break;
			} else {
				++irev;
				nrev_t+=size_tot;
				rev = rev_t[nrev_t];
			}
		} // while irev
	} // for ifwd
} // GraphReverse

} // extern "C"

