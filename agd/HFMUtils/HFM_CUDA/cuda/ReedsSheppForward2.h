#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 1; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

void scheme(const Scalar params[geom_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_XI_KAPPA_THETA(params,x)

	weights[0]=1./xi;
	offset[0][0]=0; offset[0][1]=0; offset[0][2]=1; //offset[0]={0,0,1};

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar v[ndim] = {c,s,kappa};

	Selling_v(v, &weights[1], &offset[1]);
}

#include "Update.h"