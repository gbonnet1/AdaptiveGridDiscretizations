#pragma once
#include "Geometry3.h"

const Int nsym = 0; // Number of symmetric offsets
const Int nfwd = symdim; // Number of forward offsets
const Int nmix = 2;
Scalar mix(const Scalar a, const Scalar b){return min(a,b);}
const Int metric_size = ;

void scheme(const Scalar dual_metric[metric_size], Scalar weights[nsym], Int offsets[nsym][ndim]){
	Selling_decomp(dual_metric,weights,offsets);}
