#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#define nsym_macro 0 // Only uses forward offsets
#define curvature_macro 1
#include "Geometry3.h"

/**Fejer quadrature rule for integration*/
#ifndef nFejer_macro
#define nFejer_macro 5
#endif

const Int nFejer = nFejer_macro;

// Weights used for one dimensional quadrature on [-pi/2,pi/2] with cosine weight
#if nFejer_macro==5
const Scalar wFejer[nFejer]={0.167781, 0.525552, 0.613333, 0.525552, 0.167781};
#elif nFejer_macro==9
const Scalar wFejer[nFejer]={0.0527366, 0.179189, 0.264037, 0.330845, 
	0.346384, 0.330845, 0.264037, 0.179189, 0.0527366}
#endif

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = nFejer*symdim; // Number of forward offsets

#include "Constants.h"

#if precomputed_scheme_macro
// const int nTheta // Must be defined in including file
__constant__ Scalar precomp_weights[nTheta][nactx];
__constant__ Scalar precomp_offsets[nTheta][nactx][ndim];
#else
void scheme(GEOM(const Scalar params[geom_size],) Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	XI_VAR(Scalar xi;) KAPPA_VAR(Scalar kappa;) Scalar theta;
	get_xi_kappa_theta(GEOM(geom,) x, XI_VAR(xi,) KAPPA_VAR(kappa,) theta);
	const Scalar cT = cos(theta), sT = sin(theta);

	for(Int l=0; l<nFejer; ++l){
		const Scalar phi = pi*(l+0.5)/nFejer;
		const Scalar cP = cos(phi), sP = sin(phi);
		const Scalar v[ndim]={sP*cT,sP*sT,(sP*kappa+cP/xi)};

		Selling_v(v, &weights[l*symdim], &offsets[l*symdim]);
		const Scalar s = wFejer[l];
		for(int i=0; i<symdim; ++i) weights[l*symdim+i] *= s;
	}
}
#endif

#include "Update.h"