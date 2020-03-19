#pragma once
// ----------- Constants -------------

Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}


/// Tolerance for the fixed point solver.
__constant__ Scalar tol;


#if multiprecision_macro
__constant__ Scalar multip_step;
__constant__ Scalar multip_max; // Drop multi-precision beyond this value to avoid overflow
#endif


/// Shape of the outer domain
__constant__ Int shape_o[ndim];
__constant__ Int size_o;

/// Shape of the full domain
__constant__ Int shape_tot[ndim]; // shape_i * shape_o
__constant__ Int size_tot; // product(shape_tot)

#if isotropic_macro
const Int metric_size = 1;
#endif

#if factor_macro
__constant__ Scalar factor_metric[metric_size];
__constant__ Scalar factor_origin[ndim];
__constant__ Scalar factor_radius2;

// Input: absolute position of point. 
// Output: wether factor happens here, and relative position of point.
bool factor_rel(const Int x_abs[ndim], Scalar x_rel[ndim]){
	Scalar r2 = 0.;
	for(Int k=0; k<ndim; ++k){
		x_rel[k] = x_abs[k]-factor_origin[k];
		r2+=x_rel[k]*x_rel[k];
	}
	return r2 < factor_radius2;}
#endif