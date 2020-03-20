#pragma once

#include "Constants.h"
typedef Int VC; // Vector component
typedef Scalar MC; // Symmetric matrix component

/// Sum 
void add_vv(const VC x[ndim], const VC y[ndim], VC out[ndim]){
	for(Int i=0; i<ndim; ++i){
		out[i]=x[i]+y[i];}
}

/// Difference
void sub_vv(const VC x[ndim], const VC y[ndim], VC out[ndim]){
	for(Int i=0; i<ndim; ++i){
		out[i]=x[i]-y[i];}
}

/// Opposite vector
void neg_v(const VC x[ndim], VC out[ndim]){
	for(Int i=0; i<ndim; ++i){
		out[i]=-x[i];}
}

/// Perpendicular vector, in dimension two. Caution : assume x and out are distinct.
void perp_v(const VC x[2], VC out[2]){ 
	out[0]=-x[1];
	out[1]= x[0];
}

/// Cross product, in dimension three. Caution : assumes out is dstinct from x and y.
void cross_vv(const VC x[3], const VC y[3], VC out[3]){
	for(Int i=0; i<3; ++i){
		const Int j=(i+1)%3, k=(i+2)%3;
		out[i]=x[j]*y[k]-x[k]*y[j];
	}
}

/// Euclidean scalar product
template<typename Tx, typename Ty, typename Tout=Tx>
Tout scal_vv(const Tx x[ndim], const Ty y[ndim]){
	Tout result=0.;
	for(Int i=0; i<ndim; ++i){
		result+=x[i]*y[i];}
	return result;
}

/// Scalar product associated with a symmetric matrix
template<typename Tx,typename Ty>
Scalar scal_vmv(const Tx x[ndim], const MC m[symdim], const Ty y[ndim]){
	Scalar result=0;
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			result += (i==j ? 1 : 2)*x[i]*y[j]*m[k]; 
			++k;
		}
	}
	return result;
}

void canonicalsuperbase(VC sb[ndim+1][ndim]){
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<ndim; ++j){
			sb[i][j]= (i==j);
		}
	}
	for(Int j=0; j<ndim; ++j){
		sb[ndim][j]=-1;
	}
}