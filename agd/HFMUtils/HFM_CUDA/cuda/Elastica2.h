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

// Fejer weights are for one dimensional quadrature on [-pi/2,pi/2] with cosine weight
// Array suffix _s indicates stencil data.
//def cos_sin_table(n): // Generate the cosine and sine tables
//    angles = (np.arange(n)+0.5)*np.pi/n
//    return np.cos(angles),np.sin(angles)
#if nFejer_macro==5
const Scalar wFejer_s[nFejer]={0.167781, 0.525552, 0.613333, 0.525552, 0.167781};
const Scalar cosPhi_s[nFejer]={0.951056516, 0.587785252, 0., -0.587785252, -0.951056516};
const Scalar sinPhi_s[nFejer]={0.309016994, 0.809016994, 1., 0.809016994, 0.309016994};

#elif nFejer_macro==9
const Scalar wFejer_s[nFejer]={0.0527366, 0.179189, 0.264037, 0.330845, 
const Scalar cosPhi_s[nFejer]={0.984807753, 0.866025404, 0.64278761, 0.342020143, 0.,
	-0.342020143, -0.64278761, -0.866025404, -0.984807753};
const Scalar sinPhi_s[nFejer]={0.173648178,   0.5, 0.766044443, 0.939692621, 1., 
	0.939692621, 0.766044443,   0.5, 0.173648178};
#endif

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = nFejer*symdim; // Number of forward offsets

#include "Constants.h"

#if precomputed_scheme_macro
void scheme(GEOM(const Scalar params[geom_size],) Int x[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	XI_VAR(Scalar ixi;) KAPPA_VAR(Scalar kappa;) 
	Scalar cT, sT; // cos(theta), sin(theta)
	get_ixi_kappa_theta(GEOM(geom,) x, XI_VAR(ixi,) KAPPA_VAR(kappa,) cT, sT);

	for(Int l=0; l<nFejer; ++l){
//		const Scalar phi = pi*(l+0.5)/nFejer; // Now using tables for speed
//		const Scalar cP = cos(phi), sP = sin(phi);
		const Scalar cP = cosPhi_s[l], sP = sinPhi_s[l];
		const Scalar v[ndim]={sP*cT,sP*sT,(sP*kappa+cP*ixi)};

		Selling_v(v, &weights[l*symdim], &offsets[l*symdim]);
		const Scalar s = wFejer_s[l];
		for(int i=0; i<symdim; ++i) weights[l*symdim+i] *= s;
	}
}
#endif

#include "Update.h"