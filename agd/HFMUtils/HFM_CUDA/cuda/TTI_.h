#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements numerical scheme for a class of Finslerian eikonal,
known as tilted transversally isotropic, and arising in seismology.

 The dual unit ball is defined by
 < linear,p > + (1/2)< p,quadratic,p > = 1
 where p is the vector containing the squares of transform*p0.

 This code is mostly adapted from the HamiltonFastMarching cpu eikonal solver.
*/

#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#endif

namespace dim2 { // Some two dimensional linear algebra is needed in any case.
	const Int ndim = 2;
	#include "Geometry_.h"
}

// Number of schemes of which to take the minimum or maximum.
#ifndef nmix_macro
const Int nmix = 10;
#endif

const Int nsym = symdim;
const Int nfwd = 0;
const Int geom_size = dim2::ndim + dim2::symdim + ndim*ndim;
// const Int factor_size = ??; To be determined


#include "Constants.h"

namespace dim2 { // ndim=2, symdim=3

Scalar det_m(const Scalar m[symdim]){
	return coef_m(m,0,0)*coef_m(m,1,1)-coef_m(m,0,1)*coef_m(m,1,0);}
Scalar det_vv(const Scalar x[ndim], const Scalar y[ndim]){
	return x[0]*y[1]-x[1]*y[0];}

/// Returns smallest positive s such that 0.5*q*s^2 + l*s -1 = 0
Scalar solve(const Scalar l, const Scalar q){
	if(q==0){return 1/l;}
	const ScalarType delta = l*l + 2.*q;
	const ScalarType sdelta = sqrt(delta);
	const ScalarType rm = (- l - sdelta)/q, rp = (- l + sdelta)/q;
	const ScalarType rmin=min(rm,rp), rmax=max(rm,rp);
	return rmin>0 ? rmin : rmax;
}

Scalar slope(const Scalar x[ndim]){return x[1]/(x[0]+x[1]);}

/// Computes the extremal slopes and mix_is_min
bool properties(const Scalar linear[ndim], const Scalar quadratic[symdim],
	Scalar slopes[2]){

	const Scalar root0 = solve(linear[0],coef_m(quadratic,0,0));
	const Scalar root1 = solve(linear[1],coef_m(quadratic,1,1));
	
	slopes[0] = slope({linear[0]+ quadratic(0,0)*root0,linear[1]+quadratic(1,0)*root0});
	slopes[1] = slope({linear[0]+ quadratic(0,1)*root1,linear[1]+quadratic(1,1)*root1});

	return slopes[0]<slopes[1]; // TODO : check
}

Scalar multiplier(const Scalar linear[ndim], const Scalar quadratic[symdim],
	const Scalar slope){ 

	const Scalar Q[symdim] = { // comatrix of quadratic
		coef_m(quadratic,1,1),-coef_m(quadratic,0,1),coef_m(quadratic,0,0)};
	const Sym2 Q = quadratic.Comatrix();
	const Scalar * l = linear; // l[2]
	Scalar Ql[ndim]; dot_mv(Q,l,Ql);
	const Scalar detQ = det_m(Q);
	const Scalar lQl = scal_vv(l,Ql);

	const Scalar v[2]={Scalar(1)-slope,slope};
	Scalar Qv[2]; dot_mv(Q,v,Qv);
	const Scalar detVL = det_vv(v,l);
	
	const Scalar lQv = scal_vv(l,Qv);
	const Scalar vQv = scal_vv(v,Qv);
	
	const Scalar num = detVL*detVL + 2*vQv;
	const int signNum = num>0 ? 1 : -1;
	
	const Scalar sdelta = sqrt(vQv*(2*detQ+lQl));
	const Scalar den = signNum * sdelta + lQv;
	return num/den;
}

}

// scheme returns mix_is_min
bool scheme(const Scalar geom[geom_size], 
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	const Scalar * linear = geom; // linear[2]
	const Scalar * quadratic = geom + 2; // quadratic[dim2::symdim]
	const Scalar * transform = geom + (2+dim2::symdim); // transform[ndim * ndim]

	Scalar slopes[2];
	const bool mix_is_min = dim2::properties(linear,quadratic,slopes);
	Scalar D0[symdim]; self_outer_v(transform,D0);
	Scalar D1[symdim]; self_outer_v(transform+ndim,D1);
	if(ndim==3){Scalar D2[symdim]; self_outer_v(transform+2*ndim,D2);
		for(Int i=0; i<symdim; ++i) D1[i]+=D2[i];}

	for(Int kmix=0; kmix<nmix; ++kmix){
		const Scalar slope = slopes[0] + (kmix/Scalar(nmix))*slopes[1];
		const Scalar mult = dim2::multiplier(linear,quadratic,slope);
		Scalar D[symdim];
		for(Int i=0; i<symdim; ++i) {
			D[i] = mult*((Scalar(1)-slope)*D0[i] + slope*D1[i]);} // TODO : check
		selling_m(D, weights+kmix*symdim, offsets + kmix*symdim);
	}
	return mix_is_min;
}


FACTOR(
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Get the optimal solves for x,x+e
	// Get the matrices
	// 
}
)
