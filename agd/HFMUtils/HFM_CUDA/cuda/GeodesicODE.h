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

typedef int Int;
typedef float Scalar;
const Int ndim = 2;

#include "Geometry_.h"

const Int ncorners = 1<<ndim;
const bool periodic[ndim]={false,false};
__constant__ shape_tot[ndim];
__constant__ size_tot[ndim];

typedef unsigned char uchar;
const uchar uchar_MAX = 255;

__constant__ Int max_len = 200; // Max geodesic length
__constant__ Scalar causalityTolerance = 4; 
__constant__ Scalar geodesicStep = 6.*sqrt(ndim);
const Int hlen = 20; // History length (a small periodic history of computations is kept)
const Int eucl_delay = hlen-1; // Used in PastSeed stopping  criterion
const Int nymin_delay = hlen-1; // Used in Stationnary stropping criterion


enum class ODEStop {
	Continue = 0, // Do not stop here
	Seed, // Correct termination
	Wall, // Went out of domain
	Stationnary, // Error : Stall in ODE process
	PastSeed, // Error : Moving away from target
	VanishingFlow, // Error : Vanishing flow
};

/** Array suffix conventions:
- t : global field [physical dims][shape_tot]
- s : data shared by all ODE solver threads [nThreads][len][physical dims]
- p : periodic buffer, for a given thread. [min_len][...]
- no suffix : basic thread dependent data.
*/

/** Computes the floor of the scalar components. Returns wether value changed.*/
bool Floor(const Scalar x[ndim], Int xq[ndim]){
	bool changed = false;
	for(Int k=0; k<ndim; ++k){
		const Int xqi = round(x[i]);
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
 - exclude_cache : corners excluded from averaging.

 Returned value : 
 - stop : if true, the minimal neighbor is a degenerate point (seed or wall)
*/
ODEStop NormalizedFlow(
	const Scalar* flow_vector_t,const Scalar* flow_weightsum_t,const Scalar* dist_t, 
	const Scalar x[ndim], Scalar flow[ndim],
	Int xq[ndim], Int * nymin,
	Scalar flow_cache[ncorners][ndim], bool exclude_cache[ncorners] ){

	ODEStop result = ODEStop::Continue;

	if(Floor(x,xq)){
		Scalar dist[ncorners];
		Scalar dist_min = infinity(); // Minimal distance among corners
		for(Int icorner=0; icorner< ncorners; ++icorner){

			// Get the i-th corner and its index in the total shape.
			Int yq[ndim]; 
			for(Int k=0; k<ndim; ++k){yq[k] = xq[k]+((icorner >> k) & 1);}
			if(!InRange_per(yq,shape_tot)){
				exclude_cache[icorner]=true; 
				dist[icorner]=infinity(); 
				continue;}
			const Int ny = Index_per(yq,shape_tot);

			// Update the minimal distance, and corresponding weightsum, and eucl distance.
			dist[icorner] = dist_t[ny];
			if(dist[icorner]<dist_min){
				dist_min=dist[icorner];
				*nymin = ny;
			}

			// Get the flow components
			for(Int k=0; k<ndim; ++k){
				flow_cache[icorner][k] = flow_vector_t[ny+size_tot*k];}
		}

		const Scalar flow_weightsum = flow_weightsum_t[*nymin];
		if(dist_min==infinity()){ODEStop=ODEStop::Wall;}
		else if(flow_weightsum==0.){ODEStop=ODEStop::Seed;}

		// Exclude interpolation neighbors with too high value.
		const Scalar dist_threshold = dist_min+causalityTolerance/flow_weightsum;
		for(Int icorner=0; icorner<ncorners; ++icorner){
			exclude_cache[icorner] = dist[icorner] < dist_threshold;}
	}

	// Perform the interpolation
//	Scalar wsum=0.;
	fill_kV(0.,flow)

	for(Int icorner=0; icorner<ncorners; ++icorner){
		// Get the corner bilinear interpolation weight.
		if(exclude_cache[icorner]) continue;
		Scalar w = 1.;
		for(Int k=0; k<ndim; ++k){
			const Scalar dxk = x[k] - xq[k]
			w *= ((icorner>>k) & 1) ? 1-dxk : dxk;
		}

		// Add corner contribution
//		wsum+=w;
		madd_kvV(w,flow_cache[icorner],flow);
	}
	
//	if(wsum>0){for(Int k=0; k<ndim; ++k){flow[k] /= wsum;}}
	const Scalar flow_norm = sqrt(scal_vv(flow));
	if(flow_norm>0){div_Vk(flow,flow_norm);}
	else if(result==ODEStop::Continue){result = ODEStop::VanishingFlow;}

	return result;
}


/*
void Flow(const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const uchar * eucl_t, 
	const Scalar x[ndim], Int xq[ndim], 
	Scalar flow_cache[ncorners][ndim], bool exclude_cache[ncorners],
	Scalar flow[ndim], uchar * eucl){
*/

extern "C" {

__global__ void GeodesicODE(
	const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const uchar * eucl_t,
	Scalar * x_s, Int * len_s, uchar * stop_s){

	const Int tid = BlockIdx.x * BlockDim.x + ThreadIdx.x;

	// Get the position, and euclidean distance to target, of the previous points
	/*
	Scalar x_p[min_len][ndim];
	const Int q_s_shape[3] = {BlockDim.x*GridDim.x, max_len, ndim};
	for(Int k=0; k<ndim; ++k){
		for(Int l=0; l<min_len; ++l){
			const Int q_s_pos[3]={tid,l,k};
			x_p[l][k] = x_s[Index(q_s_pos,q_s_shape)];
		}
	}
	*/
	// Short term periodic history introduced to avoid stalls or moving past the seed.
	uchar eucl_p[hlen];
	Int nymin_p[hlen];
	for(Int l=0; l<hlen; ++l){
		eucl_p[l]  = uchar_MAX;
		nymin_p[l] = Int_MAX;
	}

	Scalar x[ndim]; copy_vV(x_s+tid*max_len*ndim,x);
	Int xq[ndim]; fill_kV(Int_MAX,xq);
	Int nymin = Int_MAX;
	Scalar flow_cache[ncorners][ndim]; 
	bool exclude_cache[ncorners];

	Int len;
	ODEStop stop = ODEStop::Continue;
	for(len = 1; len<max_len; ++len){
		const Int l = len%hlen;

		// Compute the flow at the current position
		Scalar flow[ndim];
		stop = NormalizedFlow(
			flow_vector_t,flow_weightsum_t,dist_t,
			x,flow,
			xq,&nymin,
			flow_cache,exclude_cache)
		if(stop!=ODEStop::Continue){break;}

		// Check PastSeed and Stationnary stopping criteria
		nymin_p[l] = nymin;
		eucl_p[l] = eucl_t[nymin];

		if(nymin     == nymin_p[(l-nymin_delay+hlen)%hlen]){
			stop = ODEStop::Stationnary; break;}
		if(eucl_p[l] >  eucl_p[ (l-eucl_delay+hlen) %hlen]){
			stop = ODEStop::PastSeed;    break;}

		// Make a half step, to get the Euler midpoint
		Scalar xMid[ndim];
		madd_kvv(0.5*geodesicStep,flow,x[lPrev],xMid);

		// Compute the flow at the midpoint
		stop = NormalizedFlow(
			flow_vector_t,flow_weightsum_t,dist_t,
			xMid,flow,
			xq,&nymin,
			flow_cache,exclude_cache) 
		if(stop!=ODEStop::Continue){break;}

		madd_kvv(geodesicStep,flow,x[lPrev],x[l]);
		copy_vV(x[l],x_s + (tid*max_len + len)*ndim)
	}

	len_s[tid] = len;
	stop_s[tid] = stop;
}

} // extern "C"