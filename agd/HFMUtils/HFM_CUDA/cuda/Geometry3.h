#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=3;
#include "Geometry_.h"

const Int Selling_maxiter=100;
// the first two elements of these permutations range among all possible pairs
const Int iterReducedMax = 6;
const Int Selling_permutations[iterReducedMax][ndim+1] = { 
	{0,1,2,3},{0,2,1,3},{0,3,1,2},{1,2,0,3},{1,3,0,2},{2,3,0,1}};

void obtusesuperbase(const Scalar m[symdim], Int sb[ndim+1][ndim]){
	canonicalsuperbase(sb);
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int it = iter%6; 
		const Int * perm = Selling_permutations[it];
		const Int i = perm[0], j=perm[1];
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			const Int k=perm[2], l=perm[3];
			add_vV(sb[i],sb[k]);
			add_vV(sb[i],sb[l]);
			neg_V(sb[i]);
			iterReduced=0;
		}
	}
}

// Selling decomposition of a tensor
void Selling_m(const Scalar m[symdim], Scalar weights[symdim], Int offsets[symdim][ndim]){
	Int sb[ndim+1][ndim];
	obtusesuperbase(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int * perm = Selling_permutations[r];
		const Int i=perm[0],j=perm[1],k=perm[2],l=perm[3];
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]) );
		cross_vv(sb[k],sb[l],offsets[r]);
	}
}


CURVATURE(
__constant__ Scalar Selling_v_relax = 0.01; // Relaxation parameter for Selling_v. 
__constant__ Scalar Selling_v_cosmin2 = 2./3.; // Relaxation parameter for Selling_v.

// Based on selling decomposition, with some relaxation, reorienting of offsets, and pruning of weights
void Selling_v(const Scalar v[ndim], Scalar weights[symdim], Int offsets[symdim][ndim]){

	// Build and decompose the relaxed self outer product of v
	Scalar m[symdim];
	self_outer_relax_v(v,Selling_v_relax,m);	
	Selling_m(m,weights,offsets);
	const Scalar vv = scal_vv(v,v);

	// Redirect offsets in the direction of v, and eliminate those which deviate too much.
	for(Int k=0; k<symdim; ++k){
		Int * e = offsets[k]; // e[ndim]
		const Scalar ve = scal_vv(v,e), ee = scal_vv(e,e);
		if(ve*ve < vv*ee*Selling_v_cosmin2){weights[k]=0; continue;}
		if(ve>0){neg_V(e);} // Note : we want ve<0.
	}
}
)