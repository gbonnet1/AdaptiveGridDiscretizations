#pragma once

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = symdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets

void scheme(const Scalar params[geom_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar v[ndim] = {c,s,kappa};

	// Build the relaxed self outer product of v
	Scalar m[symdim];
	self_outer_relax_v(v,Selling_v_relax,m);
	m[5] = max(m[5], v[2]*v[2] + 1/(xi*xi));
	Selling_m(m,weights,offsets);

	// Prune offsets which deviate too much
	const Scalar w[ndim] = {v[1],-v[0],0}; // cross product of v and {0,0,1}
	const Scalar ww = scal_vv(w,w);
	for(Int k=0; k<symdim; ++k){
		const Int * e = offsets[k]; // e[ndim]
		const Scalar we = scal_vv(w,e), ee = scal_vv(e,e);
		if(we*we >= ee*ww*(1-Selling_v_cosmin2)){
			weights[k]=0;}
	}

	const Scalar speed2=speed*speed;
	for(Int k=0; k<ntotx; ++k){weights[k]*=speed2;}
}

#include "Update.h"