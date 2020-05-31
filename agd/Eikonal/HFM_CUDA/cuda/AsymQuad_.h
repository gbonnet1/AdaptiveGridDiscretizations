#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

// ndim_macro must be defined
#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#elif (ndim_macro == 4)
#include "Geometry4.h"
#elif (ndim_macro == 5)
#include "Geometry5.h"
#endif

const Int nsym = decompdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets
const Int geom_size = symdim;
const Int factor_size = symdim;

/* The asymmetric quadratic scheme reads max(min(a,b),c), 
where a is a Rander scheme, and b,c are Riemann schemes*/
#define nmix_macro 3 
#define alternating_mix_macro 1
const bool mix_is_min[3] = {true,true,false}; // First element doesn't really count

// nactx = nmix*nsym

#include "Constants.h"

void scheme(const Scalar geom[geom_size], 
	Scalar weights[nactx], OffsetT offsets[nactx][ndim], DRIFT(Scalar drift[3][ndim]) ){
	const Scalar * m = geom; // m[symdim]
	const Scalar * eta = geom+symdim; // eta[ndim]
	Scalar w[ndim]; dot_mv(m,eta,w);
	Scalar wwT[symdim]; self_outer_v(w,wwT);

	decomp_m(m,weights+nact,offsets+nact); zero_V(drift[1]);

	Scalar mwwT[symdim]; add_mm(m,wwT,mwwT);
	decomp_m(mwwT,weights+2*nact,offsets+2*nact); zero_V(drift[2]);

	// Computes an ellipse in between the two halves
	const Scalar n2 = scal_vv(w,eta); // | w |_{M^{-1}}
	const Scalar n = sqrt(n2);
	const Scalar in2 = sqrt(1.+n2);
	const Scalar iin2 = 1.+in2;
	const Scalar iin2_2 = iin2*iin2;
	const Scalar iin2_3 = iin2*iin2_2;
	const Scalar lambda = n/(2.*in2*iin2);
	const Scalar mu = 4.*in2/iin2_3;
	const Scalar gamma = 4.*(1.+n2)/iin2_2 - n2*mu;

	for(Int i=0; i<symdim; ++i){mwwT[i] = gamma*m[i]+mu*wwT[i];}
	decomp_m(mwwT,weights,offsets);
	mul_kv(lambda,eta,drift[0]);
}

FACTOR(
#include "EuclideanFactor.h"
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products and norms
	const Scalar * m = factor_metric;
	const Scalar xx=scal_vmv(x,m,x), xe=scal_vmv(x,m,e), ee=scal_vmv(e,m,e);
	euclidean_factor_sym(xx,xe,ee,fact ORDER2(, fact2));
}
) 

#include "Update.h"