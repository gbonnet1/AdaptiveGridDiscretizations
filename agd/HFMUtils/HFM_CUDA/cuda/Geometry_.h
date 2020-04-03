#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

const Int symdim = (ndim*(ndim+1))/2; // Dimension of the space of symmetric matrices.

/**
Naming conventions : 
- k : scalar input
- v : vector input
- m : symmetric matrix input
- upper case : output (terminal output may be omitted)
*/

template<typename T>
void copy_vV(const T x[ndim], T out[ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=x[i];}} 

/// Sum 
void add_vv(const Int x[ndim], const Int y[ndim], Int out[ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=x[i]+y[i];}}

void add_vV(const Int x[ndim], Int y[ndim]){
	for(Int i=0; i<ndim; ++i){y[i]+=x[i];}}

/// Difference
void sub_vv(const Int x[ndim], const Int y[ndim], Int out[ndim]){
	for(Int i=0; i<ndim; ++i){
		out[i]=x[i]-y[i];}
}

/// Opposite vector
void neg_v(const Int x[ndim], Int out[ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=-x[i];}}
void neg_V(Int x[ndim]){
	for(Int i=0; i<ndim; ++i){x[i]=-x[i];}}

/// Perpendicular vector, in dimension two. Caution : assume x and out are distinct.
void perp_v(const Int x[2], Int out[2]){ 
	out[0]=-x[1];
	out[1]= x[0];
}

template<typename T>
void fill_kV(const T k, T v[ndim]){
	for(Int i=0; i<ndim; ++i){v[i]=k;}}

template<typename T>
void mul_kV(const T k, T v[ndim]){
	for(Int i=0; i<ndim; ++i){v[i]*=k;}}

void div_Vk(Scalar v[ndim], const Scalar k){
	const Scalar l=1./k; mul_kV(l,v);}

template<typename T>
void madd_kvv(const T k, T x[ndim], const T y[ndim], T out[ndim]){
	for(Int i=0; i<ndim; ++i){out[i]=k*x[i]+y[i];}}

template<typename T>
void madd_kvV(const T k, T x[ndim], T y[ndim]){
	for(Int i=0; i<ndim; ++i){y[i]+=k*x[i];} }

/// Cross product, in dimension three. Caution : assumes out is dstinct from x and y.
template<typename Tx, typename Ty, typename Tout=Tx>
void cross_vv(const Tx x[3], const Ty y[3], Tout out[3]){
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
	Scalar result=0.;
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			result += (i==j ? x[i]*y[i] : (x[i]*y[j]+x[j]*y[i]))*m[k]; 
			++k;
		}
	}
	return result;
}



void self_outer_v(const Scalar x[ndim], Scalar m[ndim]){
	Int k=0; 
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			m[k] = x[i]*x[j]; 
			++k;
		}
	}
}

void self_outer_relax_v(const Scalar x[ndim], const Scalar relax, Scalar m[ndim]){
	const Scalar eps = scal_vv(x,x)*relax;
	Int k=0;
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<=i; ++j){
			m[k] = x[i]*x[j]*(1-eps) + (i==j)*eps; 
			++k;
		}
	}
}



void canonicalsuperbase(Int sb[ndim+1][ndim]){
	for(Int i=0; i<ndim; ++i){
		for(Int j=0; j<ndim; ++j){
			sb[i][j]= (i==j);
		}
	}
	for(Int j=0; j<ndim; ++j){
		sb[ndim][j]=-1;
	}
}