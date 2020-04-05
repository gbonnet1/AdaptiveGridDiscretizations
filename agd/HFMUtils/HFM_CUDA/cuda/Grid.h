#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements common facilities for bounds checking and array access.*/
namespace Grid {

#ifdef bilevel_grid_macro // Only if shape_tot, shape_o and shape_i are defined
Int Index_tot(const Int x[ndim]){
	// Get the index of a point in the full array.
	// No bounds check 
	Int n_o=0,n_i=0;
	for(Int k=0; k<ndim; ++k){
		Int xk=x[k];
		PERIODIC(if(periodic_axes[k]){xk = (xk+shape_tot[k])%shape_tot[k];})
		const Int 
		s_i = shape_i[k],
		x_o= xk/s_i,
		x_i= xk%s_i;
		if(k>0) {n_o*=shape_o[k]; n_i*=s_i;}
		n_o+=x_o; n_i+=x_i; 
	}
	const Int n=n_o*size_i+n_i;
	return n;
}
#endif

bool InRange(const Int x[ndim], const Int shape_[ndim]){
	for(int k=0; k<ndim; ++k){
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index(const Int x[ndim], const Int shape_[ndim]){
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		n+=x[k];
	}
	return n;
}

bool InRange_per(const Int x[ndim], const Int shape_[ndim]){
for(int k=0; k<ndim; ++k){
		PERIODIC(if(periodic_axes[k]) continue;)
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index_per(const Int x[ndim], const Int shape_[ndim]){
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		Int xk=x[k];
		PERIODIC(if(periodic_axes[k]){xk=(xk+shape_[k])%shape_[k];})
		n+=xk;
	}
	return n;
}


void Position(Int n, const Int shape_[ndim], Int x[ndim]){
	for(Int k=ndim-1; k>=1; --k){
		x[k] = n % shape_[k];
		n /= shape_[k];
	}
	x[0] = n;
}

}



