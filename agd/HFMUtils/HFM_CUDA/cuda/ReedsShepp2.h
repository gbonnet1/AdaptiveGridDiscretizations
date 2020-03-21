#pragma once

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = symdim; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets

void scheme(const Scalar params[metric_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar v[ndim] = {c,s,kappa};

	Scalar m[symdim];
	self_outer_relax(v,curv_relax,m);
	m[symdim-1]+=1./(xi*xi);

	Selling_m(m,weights,offsets);

	const Scalar speed2=speed*speed;
	for(Int k=0; k<ntotx; ++k){weights[k]*=speed2;}
}

#include "Update.h"