#pragma once

#include "TypeTraits.h"
const Int ndim=3;
#include "Geometry.h"

// the first two elements of these permutations range among all possible pairs
const Int Selling_permutations[iterReducedMax][ndim+1] = { 
	{0,1,2,3},{0,2,1,3},{0,3,1,2},{1,2,0,3},{1,3,0,2},{2,3,0,1}};

void obtusesuperbase(const Scalar m[symdim], Int sb[ndim+1][ndim]){
	canonicalsuperbase(sb);
	const Int iterReducedMax = 6;
	for(Int iter=0, iterReduced=0; 
		iter<Selling_maxiter && iterReduced < iterReducedMax; 
		++iter, ++iterReduced){
		const Int it = iter%6; 
		const Int * perm = Selling_permutations[it];
		const Int i = perm[0], j=perm[1];
		if(scal_vmv(sb[i],m,sb[j]) > 0){
			const Int k=perm[2], l=perm[3];
			add_vv(sb[i],sb[k],sb[k]);
			add_vv(sb[i],sb[l],sb[l]);
			neg_v(sb[i],sb[i]);
			iterReduced=0;
		}
	}
}

void Selling_decomp(const Scalar m[symdim], Scalar weights[symdim], Int offsets[symdim][ndim]){
	Int sb[ndim+1][ndim];
	obtusesuperbase(m,sb);
	for(Int r=0; r<symdim; ++r){
		const Int * perm = Selling_permutations[r];
		const Int i=perm[0],j=perm[1],k=perm[2],l=perm[3];
		weights[r] = max(0., - scal_vmv(sb[i],m,sb[j]) );
		offsets[r] = cross_vv(sb[k],sb[l]);
	}
}