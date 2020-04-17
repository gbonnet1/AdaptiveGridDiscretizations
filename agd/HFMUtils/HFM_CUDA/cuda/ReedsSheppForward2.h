#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 1; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

#include "Constants.h"

void scheme(GEOM(const Scalar params[geom_size],) Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	XI_VAR(Scalar xi;) KAPPA_VAR(Scalar kappa;) Scalar theta;
	get_xi_kappa_theta(GEOM(geom,) x, XI_VAR(xi,) KAPPA_VAR(kappa,) theta);

	weights[0]=1./(xi*xi);
	Int * offset = offsets[0];
	offset[0]=0; offset[1]=0; offset[2]=1; // offsets[0]={0,0,1};

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar v[ndim] = {c,s,kappa};

	Selling_v(v, &weights[1], &offsets[1]);
}

#include "Update.h"