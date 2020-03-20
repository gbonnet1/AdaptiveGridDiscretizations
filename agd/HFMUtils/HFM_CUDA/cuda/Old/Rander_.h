#pragma once

const Int nsym = symdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int metric_size = symdim + ndim;

void scheme(const Scalar dual_metric[metric_size], Scalar weights[nsym], 
	Scalar drift[nsym], Int offsets[nsym][ndim]){

	// Eikonal equation reads |grad u - w|_m = 1
	const Scalar * m = dual_metric; // m[symdim]
	const Scalar * w = dual_metric + symdim; // omega[ndim]

	Selling_decomp(m,weights,offsets);
	
	for(Int i=0; i<nsym; ++i){
		drift[i] = scal_vv(w,offsets[i]);}
}

/* The correction associated with Rander source factorization is identical to the Riemannian one.
Indeed, the difference between these two sources is a linear function. */
#if factor_macro
#include "RiemannFactor.h"
#else
// We take advantage of the factorization related variables to introduce asymmetry.
// Dummy perturbations
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	for(Int i=0; i<2; ++i){fact[i]=0; ORDER2(fact2[i]=0;)}
}
#undef FACTOR
#define FACTOR(...) __VA_ARGS__
#endif

#include "Update.h"