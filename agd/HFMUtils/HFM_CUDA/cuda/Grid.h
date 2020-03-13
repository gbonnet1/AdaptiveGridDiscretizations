#pragma once
/** This file implements common facilities for bounds checking and array access.*/

bool InRange(Int x[ndim], const Int shape_[ndim]){
	for(int k=0; k<ndim; ++k){
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index(Int x[ndim], const Int shape_i[ndim], const Int shape_o[ndim]) const {
	// Get the index of a point in the full array.
	// No bounds check 
	Int n_o=0,n_i=0;
	for(Int k=0; k<ndim; ++k){
		const Int 
		s_i = shape_i[k],
		x_o= x[k]/s_i,
		x_i= x[k]%s_i;
		if(k>0) {n_o*=shape_o[k]; n_i*=s_i;}
		n_o+=x_o; n_i+=x_i; 
	}

	const Int n=n_o*size_i+n_i;
	return n;
}

Int Index(Int x[ndim], const Int shape_[ndim]) const {
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		n+=x[k];
	}
	return n;
}

bool GetBool(const BoolPack * arr, const Int n){
	const Int m = 8*sizeof(BoolPack);
	const Int q = n/m, r=n%m;
	return (arr[q] >> r) & BoolPack(1);
}
