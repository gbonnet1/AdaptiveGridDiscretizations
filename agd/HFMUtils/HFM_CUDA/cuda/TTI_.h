#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/** This file implements numerical scheme for a class of Finslerian eikonal,
known as tilted transversally isotropic, and arising in seismology.

 The dual unit ball is defined by
 < linear,p > + (1/2)< p,quadratic,p > = 1
 where p is the vector containing the squares of transform*p0.
*/

/** The following constant must be defined in including file.
// Number of schemes of which to take the minimum or maximum.
const Int nmix
*/

#if (ndim_macro == 2)
#include "Geometry2.h"
#elif (ndim_macro == 3)
#include "Geometry3.h"
#endif

namespace dim2 { // Some two dimensional linear algebra is needed, even in dimension 3.
	const Int ndim = 2;
	#include "Geometry_.h"
}

const Int nsym = symdim;
const Int nfwd = 0;
const Int geom_size = dim2::ndim + dim2::symdim + ndim*ndim;

// For factorization
const Int factor_size = geom_size;
const Int niter_golden_search = 6;

#include "Constants.h"

/*Returns the two roots of a quadratic equation, a + 2 b t + c t^2.
The discriminant must be non-negative, but aside from that the equation may be degenerate*/
void solve2(const Scalar a, const Scalar b, const Scalar c, Scalar r[2]){
	const Scalar sdelta = sqrt(b*b-a*c);
	const Scalar u = - b + sdelta, v = -sdelta-b;
	if(abs(c)>abs(a)) {r[0] = u/c; r[1] = v/c;        return;}
	else if(a!=0)     {r[0] = a/u; r[1] = a/v;        return;}
	else              {r[0] = 0;   r[1] = infinity(); return;}
}

/*Returns the smallest root of the considered quadratic equation above the given threshold.
Such a root is assumed to exist.*/
Scalar solve2_above(const Scalar a, const Scalar b, const Scalar c, const Scalar above){
	Scalar r[2]; solve2(a,b,c,r);
	const bool ordered = r[0]<r[1];
	const Scalar rmin = ordered ? r[0] : r[1], rmax = ordered ? r[1] : r[0];
	return rmin>=above ? rmin : rmax;
}

namespace dim2 {


/*Scalar det_m(const Scalar m[symdim]){
	return coef_m(m,0,0)*coef_m(m,1,1)-coef_m(m,0,1)*coef_m(m,1,0);}*/
Scalar det_vv(const Scalar x[ndim], const Scalar y[ndim]){
	return x[0]*y[1]-x[1]*y[0];}

/** Returns df(x)/<x,df(x)> where f(x):= C + 2 <l,x> + <qx,x> */
void grad_ratio(const Scalar l[2], const Scalar q[3], const Scalar x[2], Scalar g[2]){
		Scalar hgrad[2]; dot_mv(q,x,hgrad); add_vV(l,hgrad); // df(x)/2
		const Scalar xhg = scal_vv(x,hgrad);
		g[0]=hgrad[0]/xhg; g[1]=hgrad[1]/xhg;
}

/** Samples the curve defined by f(x)=0,x>=0, 
where f(x):= -2 + 2 <l,x> + <qx,x>,
and returns diag(i) := .*/
bool diags(const Scalar l[2], const Scalar q[3], Scalar diag_s[nmix][2]){
	// Equation is <l,x> + 0.5 <x,q,x> = 1
	const Scalar a = solve2_above(-2,l[0],q[0],0.); // (a,0) is on the curve
	const Scalar b = solve2_above(-2,l[1],q[2],0.); // (0,b) is on the curve

	// Change of basis 
	const Scalar e0[2] = {1/2.,1/2.}, e1[2] = {1/2.,-1/2.};
	const Scalar L[2] = {scal_vv(l,e0),scal_vv(l,e1)};
	const Scalar Q[3] = {scal_vmv(e0,q,e0),scal_vmv(e0,q,e1),scal_vmv(e1,q,e1)};

	Scalar x_s[nmix][2]; // Curve sampling
	Scalar * xbeg = x_s[0], * xend = x_s[nmix-1];
	xbeg[0]=a; xbeg[1]=0; xend[0]=0; xend[1]=b;
	for(Int i=1;i<nmix-1; ++i){
		const Scalar t = i/Scalar(nmix-1);
		const Scalar v = (1-t)*a - t*b;
		// Solving f(u e0+ v e_1) = 0 w.r.t u
		const Scalar u = solve2_above(-2+2*L[1]*v+Q[2]*v*v,
			L[0]+Q[1]*v,Q[0],abs(v));
		// Inverse change of basis
		Scalar * x = x_s[i];
		x[0] = (u+v)/2; x[1] = (u-v)/2;
	} 
	for(Int i=0; i<nmix; ++i){grad_ratio(l,q,x_s[i],diag_s[i]);}

/*	if(debug_print && threadIdx.x==0){
		printf("niter_i %i",niter_i);
		printf("SetDiags l=%f,%f, q=%f,%f,%f\n",l[0],l[1],q[0],q[1],q[2]);
		printf("a=%f,b=%f\n",a,b);
		printf("L %f,%f\n",L[0],L[1]);
		const bool mix_is_min = det_vv(diag_s[0],diag_s[nmix-1])>0;
		printf("mix_is_min %i, mix_neutral %f\n",mix_is_min,mix_neutral(mix_is_min));
		printf("det_vv(diag_s[0],diag_s[nmix-1]) %f\n",det_vv(diag_s[0],diag_s[nmix-1]));
		for(Int i=0.; i<nmix; ++i){
			printf("x_s %f,%f ",x_s[i][0],x_s[i][1]);
			printf("diag_s %f,%f ",diag_s[i][0],diag_s[i][1]);
			printf("f(x_s[i]) %f \n",2*scal_vv(l,x_s[i])+scal_vmv(x_s[i],q,x_s[i]));
		}
	}*/

	return det_vv(diag_s[0],diag_s[nmix-1])>0;
}

// returns the position where value needs to be updated
Int golden_search(Scalar x[2], Scalar v[2], const bool mix_is_min){
	const Scalar phi = (sqrt(5.)-1.)/2, psi = 1.-phi, delta = psi*psi/(phi-psi);
	if(mix_is_min == (v[0]<v[1])){
		x[0]-=(x[1]-x[0])*delta
		x[1]=x[0];
		v[1]=v[0];
		return 0; 
	} else {
		x[1]+=(x[1]-x[0])*delta;
		x[0]=x[1];
		v[0]=v[1];
		return 0.;
	}
}

Scalar tti_norm(const Scalar l[2], const Scalar q[3], const Scalar s[2], Scalar diag[2]){
	const Scalar a = solve2_above(-2,l[0],q[0],0.); // (a,0) is on the curve
	const Scalar b = solve2_above(-2,l[1],q[2],0.); // (0,b) is on the curve

	// Change of basis 
	const Scalar e0[2] = {1/2.,1/2.}, e1[2] = {1/2.,-1/2.};
	const Scalar L[2] = {scal_vv(l,e0),scal_vv(l,e1)};
	const Scalar Q[3] = {scal_vmv(e0,q,e0),scal_vmv(e0,q,e1),scal_vmv(e1,q,e1)};

	const Scalar phi = (sqrt(5.)-1.)/2, psi = 1.-phi;
	Scalar x = {phi*a+phi*(-b), ...} // Set correct positions and values
	for(Int i=0; i<niter_golden_search; ++i){
		grad_ratio(l,q,x,diag); scal_vv(diag,s); // Update required value
	}
	//return minimal value, and diag
}

} // namespace dim2

void dot_av(const Scalar a[ndim][ndim], const Scalar x[ndim], Scalar out[ndim]){
	for(Int i=0; i<ndim; ++i) out[i] = dot_vv(A[i],x);}
void tdot_av(const Scalar a[ndim][ndim], const Scalar x[ndim], Scalar out[ndim]){
	fill_kV(Scalar(0),out);
	for(Int i=0; i<ndim; ++i) {for(Int j=0; j<ndim; ++j) out[i] += A[j][i]*x[j];}}

Scalar tti_norm(const Scalar l[2], const Scalar q[3], 
	const Scalar A[ndim][ndim], const Scalar x[ndim], Scalar * gradient=nullptr){
	Scalar Ax[ndim]; dot_av(A,x,Ax);
	Scalar s[2] = {Ax[0]*Ax[0],Ax[1]*Ax[1];}
	if(ndim==3) {s[1]+=Ax[ndim-1]*Ax[ndim-1];}

	const Scalar norm = tti_norm(l,q,diag);
	if(gradient!=nullptr) {
		for(Int i=0; i<ndim; ++i) {Ax[i]*=diag[min(i,2)];}
		tdot_av(A,Ax,gradient);}
	return norm;
}

bool scheme(const Scalar geom[geom_size], 
	Scalar weights[nactx], Int offsets[nactx][ndim]){
	const Scalar * linear = geom; // linear[2]
	const Scalar * quadratic = geom + 2; // quadratic[dim2::symdim]
	const Scalar * transform = geom + (2+dim2::symdim); // transform[ndim * ndim]

	Scalar diag_s[nmix][2];
	const bool mix_is_min = dim2::diags(linear,quadratic,diag_s);
	Scalar D0[symdim]; self_outer_v(transform,D0);
	Scalar D1[symdim]; self_outer_v(transform+ndim,D1);
	if(ndim==3){Scalar D2[symdim]; self_outer_v(transform+2*ndim,D2);
		for(Int i=0; i<symdim; ++i) D1[i]+=D2[i];}
	for(Int kmix=0; kmix<nmix; ++kmix){
		const Scalar * diag = diag_s[kmix]; // diag[2];
		Scalar D[symdim];
		for(Int i=0; i<symdim; ++i) {D[i] = diag[0]*D0[i] + diag[1]*D1[i];} 
		Selling_m(D, weights+kmix*symdim, offsets+kmix*symdim);
	}
	return mix_is_min;
}


FACTOR(
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Get the optimal solves for x,x+e
	// Get the matrices
	const Scalar * l = factor_metric; // linear[2]
	const Scalar * q = factor_metric + 2; // quadratic[dim2::symdim]
	const Scalar * A = factor_metric + (2+dim2::symdim); // transform[ndim * ndim]

	if(debug_print && threadIdx.x==0){
		printf("A = %f,%f,%f,%f \n",A[0],A[1],A[2],A[3]);}

	// TODO
	fact[0]=0; fact[1]=0;
}
)

#include "Update.h"