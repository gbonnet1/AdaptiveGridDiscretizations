#pragma once

#define mix_macro 1
const Int nmix = 2;
const bool mix_is_min = true;
#endif

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

void scheme(const Scalar params[metric_size],  Int x[ndim],
	Scalar weights[nmix*nfwd], Int offsets[nmix*nfwd][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar 
	vL[ndim]={c,s,kappa+1./xi},
	vR[ndim]={c,s,kappa-1./xi};
		
	Selling_v(vL,  weights,        offsets);
	Selling_v(vR, &weights[nfwd], &offsets[nfwd]);

	const Scalar speed2 = speed*speed;
	for(Int k=0; k<nmix*nfwd; ++k){
		weights[k]*=speed2;}
}

#include "Update.h"