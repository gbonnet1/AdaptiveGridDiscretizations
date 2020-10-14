#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

namespace dim3 {
#include "Geometry3.h" // Dimension used for Selling decomposition
}
const Int ndim = 5;

#if foward_macro // Reeds-Shepp forward model
const Int nsym = 2; // Isotropic metric on the sphere
const Int nfwd = dim3::symdim; // Decomposition of the 3d vector
#else // Standard Reeds-Shepp model
const Int nsym = 2+dim3::symdim;
const Int nfwd = 0;
#endif

#include "Constants.h"
__constant__ Scalar ixi; // inverse of the xi parameter, penalizing curvature 

namespace SphereProjection {
	
const Int ndim=2; // Dimension of the sphere
#include "Geometry_.h"
__constant__ Scalar h;
__constant__ Scalar r;
#if sphere_macro
__constant__ Scalar sep_r;
#endif

// Returns a three dimensional point in the unit sphere, and a conformal cost
Scalar conformal_cost(const Int x[ndim], Scalar q[ndim+1]){
	// Get the position in equator
	Scalar x_p[ndim];
	#if sphere_macro
	x_p[0] = (x[0]+0.5)*h;
	const bool first_chart = x_p[0] < 2*r+sep_r;
	x_p[0] -= first_chart ? r : (3*r+2*sep_r);
	#endif
	for(Int i=sphere_macro; i<ndim; ++i){x_p[i] = (x[i]+0.5)*h-r;}

	// Compute the conformal cost
	const Scalar s = norm2_v(x_p);
	const Scalar cost = 2./(1.+s);

	// Compute the projection of the sphere
	mul_kvV(cost,x_p,q);
	q[ndim] = (1-s)/(1+s);
	#if sphere_macro
	if(first_chart) {q[ndim] *= -1;}
	#endif
	return cost;
}

} // Sphere projection


#if !precomputed_scheme_macro
void scheme(GEOM(const Scalar geom[geom_size],)  const Int x_t[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){

	Scalar x_sphere[3];
	const Scalar cost = conformal_cost(&x_t[3],x_sphere);

	// First two weights and offsets are for the sphere geometry
	fill_kV(0,offsets[0]); offsets[0][3]=1;
	fill_kV(0,offsets[1]); offsets[1][4]=1;
	weights[0] = (ixi*ixi)/(cost*cost);
	weights[1] = weights[0];

	// Other weights and offsets are decomposition of direction
	OffsetT offsets3[dim3::symdim][3]; 

	#if dual_macro
	// Enforces motion orthogonal to x_sphere
	Scalar m[dim3::symdim];
	dim3::self_outer_v(x_sphere,m);
	const Scalar lambda = 1-decomp_v_relax;
	Int k=0; 
	for(Int i=0; i<3; ++i){
		for(Int j=0; j<=i; ++j){
			m[k] = (i==j) - decomp_v_relax*m[k];
			++k;
		}
	}
	dim3::decomp_m(m,&weights[2],offsets3);
	// Prune these offsets as in ReedsShepp2
	for(Int k=0; k<symdim; ++k){
		const Int * e = offsets3[k]; // e[3]
		const Scalar xe = dim3::scal_vv(x_sphere,e), ee = scal_vv(e,e), xx=1;//Unit vector
		if(xe * xe >= ee * xx * (1-decomp_v_cosmin2)){weights[2+k]=0;}
	}
	#else
	// Enforces motion (positively) colinear with x_sphere
	dim3::decomp_v(x_sphere, &weights[2], offsets3);
	#endif
	for(Int i=0; i<dim3::symdim; ++i){
		for(Int j=0; j<3; ++j) {offsets[2+i][j] = offsets3[i][j];}
		offsets[2+i][3]=0;
		offsets[2+i][4]=0;
	}
}
#endif // precomputed scheme