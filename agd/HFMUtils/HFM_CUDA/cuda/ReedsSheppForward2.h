#pragma once

#define curvature_macro 1
#include "Geometry3.h"

const Int nsym = 1; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets

void scheme(const Scalar params[metric_size],  Int x[ndim],
	Scalar weights[ntotx], Int offsets[ntotx][ndim]){
	GET_SPEED_XI_KAPPA_THETA(params,x)

	const Scalar c = cos(theta), s=sin(theta);
	const Scalar 
	vL[ndim]={c,s,kappa+1./xi},
	vR[ndim]={c,s,kappa-1./xi};
		
	Selling_v(vL,  weights,        offsets);
	Selling_v(vR, &weights[nfwd], &offsets[nfwd]);

	const Scalar speed2 = speed*speed;
	for(Int k=0; k<nmix*nfwd; ++k){
		weights[k]*=speed2;}

		        const ScalarType
        speed=(*pSpeed)(index),
		xi=(*pXi)(index),
		kappa= pKappa ? (*pKappa)(index) : 0.,
        gS=param.gridScale,tS=param.dependScale;
		const ScalarType theta = pTheta ? (*pTheta)(index) : index[2]*tS;
        const ScalarType c = cos(theta), s=sin(theta);
        const VectorType v{c/gS,s/gS,kappa/tS};
        
        auto & forward = stencil.forward[0];
        reduc(&forward[0], speed*v);
        
        auto & symmetric = stencil.symmetric[0];
        symmetric[0].offset = OffsetType{0,0,1};
        symmetric[0].baseWeight = square(speed/(xi*tS));

}

#include "Update.h"