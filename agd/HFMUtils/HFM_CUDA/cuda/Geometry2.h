#pragma once

#include "TypeTraits.h"
const Int ndim=2;
#include "Geometry_.h"

void obtusesuperbase(const Scalar m[symdim], Int sb[ndim+1][ndim]){
	canonicalsuperbase(sb);
	const Int iterReducedMax = 3;
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int i=iter%3, j=(iter+1)%3,k=(iter+2)%3;
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			sub_vv(sb[i],sb[j],sb[k]);
			neg_v(sb[i],sb[i]);
			iterReduced=0;
		}
	}
}

void Selling_decomp(const Scalar m[symdim], Scalar weights[symdim], Int offsets[symdim][ndim]){
	Int sb[ndim+1][ndim];
	obtusesuperbase(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int i=r, j = (r+1)%3, k=(r+2)%3;
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]));
		perp_v(sb[k],offsets[r]);
	}
}


#ifndef shape_i_macro
const Int shape_i[ndim] = {24,24}; // Shape of a single block
const Int size_i = 24*24; // Product of shape_i
const Int log2_size_i = 10; // Upper bound on log_2(size_i)
#endif

#ifndef niter_i_macro
const Int niter_i = 48;
#endif
