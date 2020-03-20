#pragma once

#define curvature_macro 1

#include "Geometry3.h"

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
	Scalar weights[nmix*nfwd], Int offsets[nmix*nfwd][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)

		const ScalarType
		speed=(*pSpeed)(index),
		xi=(*pXi)(index),
		kappa= pKappa ? (*pKappa)(index) : 0,
		gS=param.gridScale,tS=param.dependScale;
		const ScalarType theta = pTheta ? (*pTheta)(index) : index[2]*tS;
		const ScalarType cT = cos(theta), sT=sin(theta);
		
		auto & forward = stencil.forward[0];
		for(int l=0; l<nFejer; ++l){
			const ScalarType phi = mathPi*(l+0.5)/nFejer;
			const ScalarType cP = cos(phi), sP=sin(phi);
			const VectorType v{sP*cT/gS,sP*sT/gS,(sP*kappa+cP/xi)/tS};

			reduc(&forward[6*l], v*speed);
			for(int i=0; i<6; ++i) forward[6*l+i].baseWeight*=StencilElastica2<nFejer>::fejerWeights[l];
		}


	const Scalar c = cos(theta), s=sin(theta);
	const Scalar 
	vL[ndim]={c,s,kappa+1./xi},
	vR[ndim]={c,s,kappa-1./xi};
		
	Selling_v(vL,eps,  weights,	       offsets);
	Selling_v(vR,eps, &weights[nfwd], &offsets[nfwd]);

	const Scalar speed2 = speed*speed;
	for(Int k=0; k<nmix*nfwd; ++k){
		weights[k]*=speed2;}
}

#include "Update.h"