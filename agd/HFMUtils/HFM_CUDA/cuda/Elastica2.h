#pragma once

#define curvature_macro 1
#include "Geometry3.h"

/**Fejer quadrature rule for integration*/
#ifndef nFejer_macro
#define nFejer_macro 5
#endif

const Int nFejer = nFejer_macro;

#if nFejer_macro==5
const Scalar wFejer[nFejer]={0.167781, 0.525552, 0.613333, 0.525552, 0.167781};
#elif nFejer_macro==9
const Scalar wFejer[nFejer]={0.0527366, 0.179189, 0.264037, 0.330845, 0.346384, 0.330845, 0.264037, 0.179189, 0.0527366}
#endif

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = nFejer*symdim; // Number of forward offsets

void scheme(const Scalar params[metric_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)
	const Scalar cT = cos(theta), sT = sin(theta);

	for(Int l=0; l<nFejer; ++l){
		const Scalar phi = pi*(l+0.5)/nFejer;
		const Scalar cP = cos(phi), sP = sin(phi);
		const Scalar v[ndim]={sP*cT,sP*sT,(sP*kappa+cP/xi)};

		Selling_v(v, &weights[l*symdim], &offsets[l*symdim]);
		const Scalar s = speed*speed*Fejer[l];
		for(int i=0; i<symdim; ++i) weights[l*symdim+i] *= s;
	}
}

#include "Update.h"