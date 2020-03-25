#pragma once
// ----- Compile time constants ----------

const Int nact = nsym + nfwd; // maximum number of simulatneously active offsets in the scheme
const Int ntot = 2*nsym + nfwd; // Total number of offsets in the scheme

// Maximum or minimum of several schemes

#if mix_macro==0
const Int nmix = 1;
const bool mix_is_min = true; // dummy value
#endif

const Int nactx = nmix * nact;
const Int ntotx = nmix * ntot;  

const Int symdim = (ndim*(ndim+1))/2; // Dimension of the space of symmetric matrices.
const Int Selling_maxiter = ndim==2 ? 50 : 100;

#ifndef isotropic_macro
#define isotropic_macro 0
#endif

// Special treatment of isotropic metrics, whose scheme uses a single shared weight.
#if isotropic_macro 
const Int nact_ = 1;
const Int nactx_= 1;
#define ISO(...) __VA_ARGS__
#define ANISO(...)
#else 
const Int nact_ = nact;
const Int nactx_= nactx;
#define ISO(...) 
#define ANISO(...) __VA_ARGS__
#endif

Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}
Scalar mix_neutral(){return mix_is_min ? infinity() : -infinity();}
const Scalar pi = 3.14159265358979323846;

// -------- Module constants ---------

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
const Int geom_size = 1;
#endif

#if factor_macro
// geom_size == metric_size for all cases of interest
__constant__ Scalar factor_metric[geom_size]; 
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

ORDER2(
__constant__ Scalar order2_threshold;
)

// Get the parameters for curvature penalized models
#if curvature_macro 

const Int geom_size = 1 + xi_var_macro + kappa_var_macro + theta_var_macro;

#if xi_var_macro==0
__constant__ Scalar xi;
#endif

#if kappa_var_macro==0
__constant__ Scalar kappa;
#endif

const bool periodic_axes[3]={false,false,true};

#define GET_SPEED_XI_KAPPA_THETA(params,x) { \
Int k_=0;
const Scalar speed = params[k_]; ++k_;
#if xi_var_macro
const Scalar xi = params[k_]; ++k;
#endif
#if kappa_var_macro
const Scalar kappa = params[k_]; ++k;
#endif
#if theta_var_macro
const Scalar theta = params[k_]; ++k;
#else
const Scalar theta = (2.*pi*x[2])/shape_tot[2];
#endif

#endif