#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

// ----- Compile time constants ----------

const Int nact = nsym + nfwd; // maximum number of simulatneously active offsets in the scheme
const Int ntot = 2*nsym + nfwd; // Total number of offsets in the scheme

// Maximum or minimum of several schemes

#if nmix_macro==0
const Int nmix = 1;
const bool mix_is_min = true; // dummy value
#endif

const Int nactx = nmix * nact;
const Int ntotx = nmix * ntot;  

Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}
Scalar mix_neutral(const bool mix_is_min){return mix_is_min ? infinity() : -infinity();}
const Scalar pi = 3.14159265358979323846;

// -------- Module constants ---------

/// Tolerance for the fixed point solver.
__constant__ Scalar atol;
__constant__ Scalar rtol;

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

#if factor_macro
__constant__ Scalar factor_metric[factor_size]; 
__constant__ Scalar factor_origin[ndim];
__constant__ Scalar factor_radius2;

// Input: absolute position of point. 
// Output: wether factor happens here, and relative position of point.
bool factor_rel(const Int x_abs[ndim], Scalar x_rel[ndim]){
	sub_vv(x_abs,factor_origin,x_rel);
	return norm2_v(x_rel) < factor_radius2;
}
#endif

ORDER2(
__constant__ Scalar order2_threshold = 0.3;
)

// Get the parameters for curvature penalized models
#if curvature_macro 

#if xi_var_macro==0
__constant__ Scalar ixi; // inverse of the xi parameter, penalizing curvature 
#endif

#if kappa_var_macro==0
__constant__ Scalar kappa;
#endif

const bool periodic_axes[3]={false,false,true};

#if !theta_var_macro && !precomputed_scheme_macro
// const Int nTheta must be defined in including file, and equal to shape_tot[2]
__constant__ Scalar cosTheta_s[nTheta]; // cos(2*pi*i/nTheta)
__constant__ Scalar sinTheta_s[nTheta]; // sin(...)
#endif

#if precomputed_scheme_macro
// const int nTheta // Must be defined in including file
__constant__ Scalar  precomp_weights_s[nTheta][nactx];
__constant__ OffsetT precomp_offsets_s[nTheta][nactx][ndim];
#else
void get_ixi_kappa_theta(
	GEOM(const Scalar geom[geom_size],) const Int x[ndim],
	XI_VAR(Scalar & ixi,) KAPPA_VAR(Scalar & kappa,) 
	Scalar & cosTheta, Scalar & sinTheta ){
	GEOM(Int k=0;) 
	XI_VAR(ixi = geom[k]; ++k;)
	KAPPA_VAR(kappa = geom[k]; ++k;)
	#if theta_var_macro 
	cos_theta = geom[k]; ++k;
	sin_theta = geom[k]; ++k;
	#else
	const Int iTheta = x[2];
//	theta = (2.*pi*x[2])/shape_tot[2];  // Now using precomputed trigonometric tables
	cosTheta = cosTheta_s[iTheta];
	sinTheta = sinTheta_s[iTheta];
	#endif // theta_var_macro
}
#endif // precomputed_scheme_macro
#endif // curvature_macro