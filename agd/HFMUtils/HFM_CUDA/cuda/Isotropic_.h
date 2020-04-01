// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#pragma once 

#define isotropic_macro 1

const Int nsym = ndim_macro; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets

// ndim_macro must be defined
#if (ndim_macro == 2)
#include "Geometry2.h"
const Int offsets[ndim][ndim] = {{1,0},{0,1}};

#elif (ndim_macro == 3)
#include "Geometry3.h"
const Int offsets[ndim][ndim] = {{1,0,0},{0,1,0},{0,0,1}};

#else
// Not supported presently
#endif

FACTOR(
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products
	const Scalar xx=scal_vv(x,x), xe=scal_vv(x,e), ee=scal_vv(e,e);
	const Scalar Nx = sqrt(xx), // |x|
	Nxme = sqrt(xx-2*xe+ee), // |x-e|
	Nxpe = sqrt(xx+2*xe+ee); // |x+e|
	const Scalar 
	Nx_Nxme = ( 2*xe - ee)/(Nx + Nxme), // |x|-|x-e| computed in a stable way
	Nx_Nxpe = (-2*xe - ee)/(Nx + Nxpe), // |x|-|x+e| computed in a stable way
	grad_e = xe/Nx; // <e, x/|x|>
	fact[0] = -grad_e + Nx_Nxme; // <-e,x/|x|> + |x|-|x-e|
	fact[1] =  grad_e + Nx_Nxpe; // < e,x/|x|> + |x|-|x+e|

	ORDER2(
	const Scalar 
	Nxme2 = sqrt(xx-4*xe+4*ee), // |x-2e| 
	Nxpe2 = sqrt(xx+4*xe+4*ee); // |x+2e| 
	const Scalar 
	Nxme2_Nxme = (-2*xe + 3*ee)/(Nxme+Nxme2), // |x-2e|-|x-e| computed in a stable way
	Nxpe2_Nxpe = ( 2*xe + 3*ee)/(Nxpe+Nxpe2); // |x+2e|-|x+e| computed in a stable way
	fact2[0] = 2*fact[0]-(Nx_Nxme + Nxme2_Nxme); // |x|-2|x-e|+|x-2e|
	fact2[1] = 2*fact[1]-(Nx_Nxpe + Nxpe2_Nxpe); // |x|-2|x+e|+|x+2e|
	)

	for(Int k=0; k<2; ++k){
		fact[k]*=factor_metric[0];
		ORDER2(fact2[k]*=factor_metric[0];)
	}	
}
) // FACTOR

#include "Update.h"