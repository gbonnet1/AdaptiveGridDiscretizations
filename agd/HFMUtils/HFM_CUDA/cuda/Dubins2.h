#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define nmix_macro 1
const Int nmix = 2;

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

bool scheme(const Scalar params[geom_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_XI_KAPPA_THETA(params,x)

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar 
	vL[ndim]={c,s,kappa+1./xi},
	vR[ndim]={c,s,kappa-1./xi};
		
	Selling_v(vL,  weights,        offsets);
	Selling_v(vR, &weights[nfwd], &offsets[nfwd]);

	return true; // Returns mix_is_min
}

#include "Update.h"