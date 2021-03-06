#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 1; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

#include "Constants.h"
#include "decomp_v_.h"

#if !precomputed_scheme_macro
void scheme(GEOM(const Scalar geom[geom_size],) const Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	XI_VAR(Scalar ixi;) KAPPA_VAR(Scalar kappa;) 
	Scalar cT, sT; // cos(theta), sin(theta)
	get_ixi_kappa_theta(GEOM(geom,) x, XI_VAR(ixi,) KAPPA_VAR(kappa,) cT,sT);

	weights[0]=ixi*ixi;
	Int * offset = offsets[0];
	offset[0]=0; offset[1]=0; offset[2]=1; // offsets[0]={0,0,1};

	const Scalar v[ndim] = {cT,sT,kappa};
	decomp_v(v, &weights[1], &offsets[1]);
}
#endif
#include "Update.h"