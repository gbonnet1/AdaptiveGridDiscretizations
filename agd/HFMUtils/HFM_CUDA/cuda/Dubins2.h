#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define nsym_macro 0 // Only forward offsets
#define nmix_macro 1 // Maximum of a family of two schemes. 
const Int nmix = 2;

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

#include "Constants.h"

bool scheme(GEOM(const Scalar params[geom_size],) Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	XI_VAR(Scalar ixi;) KAPPA_VAR(Scalar kappa;) 
	Scalar cT, sT; // cos(theta), sin(theta)
	get_ixi_kappa_theta(GEOM(geom,) x, XI_VAR(ixi,) KAPPA_VAR(kappa,) cT,sT);

	const Scalar
	vL[ndim]={cT,sT,kappa+ixi},
	vR[ndim]={cT,sT,kappa-ixi};
		
	Selling_v(vL,  weights,        offsets);
	Selling_v(vR, &weights[nfwd], &offsets[nfwd]);

	// Maximum of a family of two schemes. -> take the minimal update among the two.
	return true; // Returns mix_is_min
}

#include "Update.h"