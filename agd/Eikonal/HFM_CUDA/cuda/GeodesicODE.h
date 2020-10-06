#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements a basic ODE solver, devoted to backtracking the minimal geodesics
using the upwind geodesic flow computed from an Eikonal solver.
It is meant to be quite similar to the GeodesicODESolver implementation in the 
HamiltonFastMarching library.

(Note : since ODE integration is inherently a sequential process, it is admitedly a bit 
silly to solve it on the GPU. We do it here because the Python code is unacceptably slow,
and to avoid relying on compiled CPU code.)
*/

#include "static_assert.h"

#ifndef Int_macro
typedef int Int;
const Int Int_Max = 2147483647;
#endif

#ifndef Scalar_macro
typedef float Scalar;
#endif
Scalar infinity(){return 1./0.;}

#ifndef EuclT_macro
typedef unsigned char EuclT;
const EuclT EuclT_Max = 255;
#endif

#ifndef ndim_macro
const Int ndim = 2;
#endif

#ifndef periodic_macro
#define PERIODIC(...) 
#else
#define PERIODIC(...) __VA_ARGS__
//const bool periodic[ndim]={false,true}; //must be defined
#endif

const Int ncorners = 1<<ndim;
__constant__ Int shape_tot[ndim];
__constant__ Int size_tot;

#define bilevel_grid_macro 
__constant__ Int shape_o[ndim]; 
__constant__ Int shape_i[ndim]; 
__constant__ Int size_i;

#include "Geometry_.h"
#include "Grid.h"

__constant__ Int nGeodesics;
__constant__ Int max_len = 200; // Max geodesic length
__constant__ Scalar causalityTolerance = 4; 
__constant__ Scalar geodesicStep = 0.25;
__constant__ Scalar weight_threshold = 0.05;

/** The following constants must be defined. 
They are related with the sensitivity of the PastSeed and Stationnary stopping criteria
respectively
const Int eucl_delay 
const Int nymin_delay
*/

// History length, used for the above delayed stopping criteria
const Int hlen = 1 + (eucl_delay<nymin_delay ? nymin_delay : eucl_delay); 

#ifndef debug_print_macro
const Int debug_print = 0;
#endif


namespace ODEStop {
enum Enum {
	Continue = 0, // Do not stop here
	AtSeed, // Correct termination
	InWall, // Went out of domain
	Stationnary, // Error : Stall in ODE process
	PastSeed, // Error : Moving away from target
	VanishingFlow, // Error : Vanishing flow
};
}
typedef char ODEStopT;

/** Array suffix conventions:
- t : global field [physical dims][shape_tot]
- s : data shared by all ODE solver threads [nThreads][len][physical dims]
- p : periodic buffer, for a given thread. [min_len][...]
- no suffix : basic thread dependent data.
*/

/** Computes the floor of the scalar components. Returns wether value changed.*/
bool Floor(const Scalar x[ndim], Int xq[ndim]){
	bool changed = false;
	for(Int i=0; i<ndim; ++i){
		const Int xqi = floor(x[i]);
		if(xqi!=xq[i]) changed=true;
		xq[i]=xqi;
	}
	return changed;
}

/** This function estimates the flow at position x, by a bilinear interpolation of 
the flow at neighboring corners. Some corners are excluded from the interpolation, if the
associated distance value is judged to large. The neighbor flow values are reloaded 
only if necessary. Also returns the euclidean distance (or other) from the best corner to 
the target.
Inputs : 
 - flow_vector_t, flow_weightsum_t, dist_t : data fields
 - x : position where the flow is requested.
Outputs :
 - flow : requested flow, normalized for unit euclidean norm.
 - xq : from Floor(x). Initialize to Int_MAX before first call.
 - nymin : index of cube corner with minimal value.
 - flow_cache : flow at the cube corners.
 - dist_cache : distance at the cube corners.
 - threshold_cache : for cube corner exclusion.

 Returned value : 
 - stop : ODE stopping criterion (if any)
*/
ODEStop::Enum NormalizedFlow(
	const Scalar * __restrict__ flow_vector_t,
	const Scalar * __restrict__ flow_weightsum_t,
	const Scalar * __restrict__ dist_t, 
	const Scalar x[ndim], Scalar flow[ndim],
	Int xq[ndim], Int & nymin, Scalar & dist_threshold,
	Scalar flow_cache[ncorners][ndim], Scalar dist_cache[ncorners]){

	ODEStop::Enum result = ODEStop::Continue;
	const bool newCell = Floor(x,xq); // Get the index of the cell containing x
	if(newCell){ // Load cell corners data (flow and dist)
		for(Int icorner=0; icorner< ncorners; ++icorner){
			// Get the i-th corner and its index in the total shape.
			Int yq[ndim]; 
			for(Int k=0; k<ndim; ++k){yq[k] = xq[k]+((icorner >> k) & 1);}
			if(!Grid::InRange_per(yq,shape_tot)){
				dist_cache[icorner]=infinity(); 
				continue;}
			const Int ny = Grid::Index_tot(yq);
			
			// Load distance and flow 
			dist_cache[icorner] = dist_t[ny];
			for(Int k=0; k<ndim; ++k){
				flow_cache[icorner][k] = flow_vector_t[ny+size_tot*k];}
		}
	}

	// Compute the bilinear weigths.
	Scalar dx[ndim]; sub_vv(x,xq,dx); 
	Scalar weights[ncorners];
	for(Int icorner=0; icorner<ncorners; ++icorner){
		weights[icorner] = 1.;
		for(Int k=0; k<ndim; ++k){
			weights[icorner]*=((icorner>>k) & 1) ? dx[k] : Scalar(1)-dx[k];}
	}
	
	// Get the point with the smallest distance, and a weight above threshold.
	Scalar dist_min=infinity();
	Int imin;
	for(Int icorner=0; icorner<ncorners; ++icorner){
		if(weights[icorner]<weight_threshold) continue;
		if(dist_cache[icorner]<dist_min) {imin=icorner; dist_min=dist_cache[icorner];}
	}

	if(dist_min==infinity()){return ODEStop::InWall;}
	Int yq[ndim]; copy_vV(xq,yq); 
	for(Int k=0; k<ndim; ++k){if((imin>>k)&1) {yq[k]+=1;}}
	const Int ny = Grid::Index_tot(yq);
	
	// Set the distance threshold
	if(ny!=nymin){
		nymin=ny;
		const Scalar flow_weightsum = flow_weightsum_t[ny];
		if(flow_weightsum==0.){result=ODEStop::AtSeed;}
		dist_threshold=dist_min+causalityTolerance/flow_weightsum;
	}


	// Perform the interpolation, and its normalization
	fill_kV(Scalar(0),flow);
	for(Int icorner=0; icorner<ncorners; ++icorner){
		if(dist_cache[icorner]>=dist_threshold) {continue;}
		madd_kvV(weights[icorner],flow_cache[icorner],flow);
	}
	// Not that a proper interpolation would require dividing by the weights sum
	// But this would be pointless here, due to the Euclidean normalization.
	const Scalar flow_norm = norm_v(flow);
	if(flow_norm>0){div_Vk(flow,flow_norm);}
	else if(result==ODEStop::Continue){result = ODEStop::VanishingFlow;}

	return result;
}

#ifdef chart_macro
#define CHART(...) __VA_ARGS__
/* 
This function jumps the current geodesic point to a prescribed position. 
It is required in the case of a manifold defined by local charts, when the geodesic leaves 
a given chart. The provided mapping should map in the interior of another chart.

When three or more local charts are used, the mapping is discontinuous. 
Optionally, this can be detected, and the average jump be replaced with a 
jump from rounded point.

// The following must be defined externally
const Int ndim_s; // Number of dimensions in mapping (first are broadcasted)
const EuclT EuclT_chart; // EuclT value when a jump is to be done
*/
const Int ncorner_s = 1<<ndim_s;
__constant__ Int size_s;

#ifndef chart_jump_variance_macro
#define chart_jump_variance_macro 0
#endif

#if chart_jump_variance_macro
__constant__ Scalar chart_jump_variance;
#endif

void ChartJump(const Scalar * mapping_s, Scalar x[ndim]){
	// Replaces x with mapped point.	

	printf("In jump n_i = %i\n",threadIdx.x);
	printf("original : %f,%f\n",x[0],x[1]);

	// Get integer and floating point part of x. Care about array broadcasting
	const Int ndim_b = ndim - ndim_s;
	Int    xq_s[ndim_s]; 
	Scalar xr_s[ndim_s];
	Scalar * x_s = x+ndim_b; // x_s[ndim_s]
	for(Int i=0; i<ndim_s; ++i){
		xq_s[i] = floor(x_s[i]); // Integer part
		xr_s[i] = x_s[i]-xq_s[i]; // Fractional part
		x_s[i] = 0; // Erase position, for future averaging
	}

	Int    yq[ndim]; // Interpolation point
	for(Int i=0; i<ndim_b; ++i) yq[i]=0; // Broadcasting first dimensions
	
	#if chart_jump_variance_macro
	Scalar mapping_sum_s[ndim_s]; // Coordinates sum
	Scalar mapping_sqs_s[ndim_s]; // Squared coordinates sum
	for(Int i=0; i<ndim_s; ++i){mapping_sum_s[i]=0; mapping_sqs_s[i]=0;}
	#endif

	for(Int icorner=0; icorner<ncorner_s; ++icorner){
		Scalar w=1.; // Interpolation weight
		for(Int i=0; i<ndim_s; ++i){
			const Int eps = (icorner>>i) & 1;
			yq[ndim_b+i] = xq_s[i] + eps;
			w *= eps ? xr_s[i] : (1.-xr_s[i]);
		}
		const Int ny_s = Grid::Index_per(yq,shape_tot); // % size_s (unnecessary)
		printf("yq %i,%i, w %f, ny_s %i\n", yq[0],yq[1],w,ny_s);
		for(Int i=0; i<ndim_s; ++i){
			const Scalar mapping = mapping_s[size_s*i+ny_s];
//			printf("mapping %f",mapping);
			x_s[i] += w * mapping;
			#if chart_jump_variance_macro
			mapping_sum_s[i]+=mapping;
			mapping_sqs_s[i]+=mapping*mapping;
			#endif
		}
	} // for icorner

	printf("mapped : %f,%f\n",x[0],x[1]);

	#if chart_jump_variance_macro 
	// Check the variance of the mapped values. If too large, use jump at rounded point.
	Scalar mapping_var = 0.;
	for(Int i=0; i<ndim_s; ++i){
		mapping_var += mapping_sqs_s[i]/ncorner_s // Variance of i-th coordinate of mapping
		 - (mapping_sum_s[i]*mapping_sum_s[i])/(ncorner_s*ncorner_s);}

	printf("mapping_var %f\n",mapping_var);
	if(mapping_var < chart_jump_variance) return;

	// Variance of mapped values is too large. Chart discontinuity detected. 
	// Using mapping from nearest point
	for(Int i=0; i<ndim_s; ++i){ yq[ndim_b+i] = xq_s[i] + (xr_s[i]>0.5 ? 1 : 0);}
	const Int ny_s = Grid::Index_per(yq,shape_tot); // % size_s (un-necessary)
	for(Int i=0; i<ndim_b; ++i){x_s[i] = mapping_s[size_s*i+ny_s];}
	#endif // chart_variance_macro
}
#else
#define CHART(...) 
#endif // chart_macro

extern "C" {

__global__ void GeodesicODE(
	const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const EuclT * eucl_t,
	CHART(const Scalar * mapping_s,)
	Scalar * x_s, Int * len_s, ODEStopT * stop_s
	){

	const Int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid>=nGeodesics) return;

	// Short term periodic history introduced to avoid stalls or moving past the seed.
	EuclT eucl_p[hlen];
	Int nymin_p[hlen];
	for(Int l=0; l<hlen; ++l){
		eucl_p[l]  = EuclT_Max;
		nymin_p[l] = Int_Max;
	}

	Scalar x[ndim]; copy_vV(x_s+tid*max_len*ndim,x);
	Int xq[ndim]; fill_kV(Int_Max,xq);
	Int nymin = Int_Max;
	Scalar flow_cache[ncorners][ndim]; 
	Scalar dist_cache[ncorners];
	Scalar dist_threshold;

	Int len;
	ODEStop::Enum stop = ODEStop::Continue;
	for(len = 1; len<max_len; ++len){
		const Int l = len%hlen;
		Scalar xPrev[ndim],xMid[ndim];
		copy_vV(x,xPrev);

		// Compute the flow at the current position
		Scalar flow[ndim];
		stop = NormalizedFlow(
			flow_vector_t,flow_weightsum_t,dist_t,
			x,flow,
			xq,nymin,dist_threshold,
			flow_cache,dist_cache);

		if(stop!=ODEStop::Continue){break;}

		// Check PastSeed and Stationnary stopping criteria
		nymin_p[l] = nymin;
		eucl_p[l] = eucl_t[nymin];
		
		#ifdef chart_macro 
		const bool chart = eucl_p[l]==EuclT_chart; // Detect wether chart change needed
		if(chart) {eucl_p[l] = eucl_p[(l-1+hlen)%hlen];} // use previous value
		#endif

		if(nymin     == nymin_p[(l-nymin_delay+hlen)%hlen]){
						stop = ODEStop::Stationnary; break;}
		if(eucl_p[l] >  eucl_p[ (l-eucl_delay+hlen) %hlen]){
			stop = ODEStop::PastSeed;    break;}

		// Make a half step, to get the Euler midpoint
		madd_kvv(Scalar(0.5)*geodesicStep,flow,xPrev,xMid);

		// Compute the flow at the midpoint
		stop = NormalizedFlow(
			flow_vector_t,flow_weightsum_t,dist_t,
			xMid,flow,
			xq,nymin,dist_threshold,
			flow_cache,dist_cache);
		if(stop!=ODEStop::Continue){break;}

		madd_kvv(geodesicStep,flow,xPrev,x);
		copy_vV(x,x_s + (tid*max_len + len)*ndim);

		CHART(if(chart) {ChartJump(mapping_s,x);})

	}

	len_s[tid] = len;
	stop_s[tid] = ODEStopT(stop);
}

} // extern "C"