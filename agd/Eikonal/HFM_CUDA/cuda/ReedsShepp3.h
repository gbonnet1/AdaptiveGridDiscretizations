#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

namespace dim3 {
#include "Geometry3.h"
}

#if foward_macro // Reeds-Shepp forward model
const Int nsym = 2; // Isotropic metric on the sphere
const Int nfwd = dim3::symdim; // Decomposition of the 3d vector
#else // Standard Reeds-Shepp model
const Int nsym = 2+dim3::symdim;
const Int nfwd = 0;
#endif

#include "Constants.h"

namespace SphereProjection {
const Int ndim=2; // Dimension of the sphere
__constant__ Scalar gridscale;

bool toplane(const Int x_t[ndim], Scalar x_p[ndim]){
//	const Int n = shape_tot[
	const bool second_chart = x_t[0]<n;
	if(second_chart) x_t[0] -= n+1;
	for(Int i=0; i<ndim; ++i){x_p[i] = (x_t[i]-0.5*n+1.)*gridscale;}
}
Scalar tocost(Scalar x_p[ndim]){
	Scalar s=0; 
	for(Int i=0; i<ndim; ++i) {s+=x_p[i]*x_p[i];}
	return 2./(1.+s);
}
Scalar tosphere(const Scalar x_p[ndim], bool second_chart, Scalar x_sp[ndim+1]){
	copy_vV(x_p,x_sp);
	Scalar s=0; 
	for(Int i=0; i<ndim; ++i) {s+=x_p[i]*x_p[i];}
	x_sp[ndim] = (1-s)/2.;
	if(second_chart) x_sp[ndim] *= -1.;
	const Scalar cost = 2./(1.+s);
	for(Int i=0; i<=ndim; ++i) x_sp[i]*cost;
	return cost;
}

}

#if !precomputed_scheme_macro
void scheme(GEOM(const Scalar geom[geom_size],)  const Int x_t[ndim],
	Scalar weights[nactx], Int offsets[nactx][ndim]){

	// Last two coordinates of x_t correspond to sphere.
	Scalar x_plane[2]; Scalar x_sphere[3]; 
	const bool second_chart  = SphereProjection::toplane(&x_t[3],x_plane);
	const Scalar sphere_cost = SphereProjection::tosphere(x_plane,second_chart,x_sphere);

	// First two weights and offsets are for the sphere geometry
	fill_kV(0,offsets[0]); offsets[0][3]=1;
	weights[0] = sphere_cost;
	fill_kV(0,offsets[1]); offsets[1][4]=1;
	weights[1] = weights[0];

	// Other weights and offsets are decomposition of direction
	
}
#endif // precomputed scheme