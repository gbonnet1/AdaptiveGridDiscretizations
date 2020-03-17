#pragma once
// ----------- Constants -------------

Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}


/// Tolerance for the fixed point solver.
__constant__ Scalar tol;


#if multi_precision_macro
__constant__ Scalar multip_step;
__constant__ Scalar multip_umax; // Drop multi-precision beyond this value to avoid overflow
#endif


/// Shape of the outer domain
__constant__ Int shape_o[ndim];
__constant__ Int size_o;

/// Shape of the full domain
__constant__ Int shape_tot[ndim]; // shape_i * shape_o
__constant__ Int size_tot; // product(shape_tot)


#if factor_macro
__constant__ Scalar factor_metric[factor_macro][metric_size];
__constant__ Scalar factor_origin[factor_macro][ndim];
#endif