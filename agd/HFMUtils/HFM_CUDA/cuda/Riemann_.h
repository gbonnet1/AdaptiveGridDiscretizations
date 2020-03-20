#pragma once

const Int nsym = symdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int metric_size = symdim;

void scheme(const Scalar dual_metric[metric_size], Scalar weights[nsym], Int offsets[nsym][ndim]){
	Selling_decomp(dual_metric,weights,offsets);}

FACTOR(
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products and norms
	const Scalar * m = factor_metric;
	const Scalar xx=scal_vmv(x,m,x), xe=scal_vmv(x,m,e), ee=scal_vmv(e,m,e);
	const Scalar Nx = sqrt(xx), // |x|_m
	Nxme = sqrt(xx-2*xe+ee), // |x-e|_m
	Nxpe = sqrt(xx+2*xe+ee); // |x+e|_m
	const Scalar 
	Nx_Nxme = ( 2*xe - ee)/(Nx + Nxme), // |x|_m-|x-e|_m computed in a stable way
	Nx_Nxpe = (-2*xe - ee)/(Nx + Nxpe), // |x|_m-|x+e|_m computed in a stable way
	grad_e = xe/Nx; // <e, m x/|x|_m>
	fact[0] = -grad_e + Nx_Nxme; // <-e,m x/|x|_m> + |x|_m-|x-e|_m
	fact[1] =  grad_e + Nx_Nxpe; // < e,m x/|x|_m> + |x|_m-|x+e|_m

	ORDER2(
	const Scalar 
	Nxme2 = sqrt(xx-4*xe+4*ee), // |x-2e|_m
	Nxpe2 = sqrt(xx+4*xe+4*ee); // |x+2e|_m
	const Scalar 
	Nxme2_Nxme = (-2*xe + 3*ee)/(Nxme+Nxme2), // |x-2e|_m-|x-e|_m computed in a stable way
	Nxpe2_Nxpe = ( 2*xe + 3*ee)/(Nxpe+Nxpe2); // |x+2e|_m-|x+e|_m computed in a stable way
	fact2[0] = 2*fact[0]-(Nx_Nxme + Nxme2_Nxme); // parenth : |x|_m-2|x-e|_m+|x-2e|_m
	fact2[1] = 2*fact[1]-(Nx_Nxpe + Nxpe2_Nxpe); // parenth : |x|_m-2|x+e|_m+|x+2e|_m
	)
}
) 

#include "Update.h"