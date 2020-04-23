#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0
/**
This file implements the of a block of values, in the HFM algorithm.
*/

#include "HFM.h"
#include "EqSeq.h"

void HFMIter(const bool active, 
	const Scalar rhs, MIX(const bool mix_is_min,) const Scalar weights[__restrict__ nactx],
	const Scalar v_o[__restrict__ ntotx], MULTIP(const Int vq_o[__restrict__ ntotx],) 
	const Int v_i[__restrict__ ntotx], 
	ORDER2(const Scalar v2_o[__restrict__ ntotx], MULTIP(const Int vq2_o[__restrict__ ntotx],) 
		const Int v2_i[__restrict__ ntotx],)
	Scalar u_i[__restrict__ size_i] MULTIP(, Int uq_i[__restrict__ size_i]) 
	FLOW(, Scalar flow_weights[__restrict__ nact] NSYM(, Int active_side[__restrict__ nsym]) 
		MIX(, Int & kmix_) ) ){
	const Int n_i = threadIdx.x;


	#if ! nmix_adaptive_macro // Simple iteration 

	for(int iter_i=0; iter_i<niter_i; ++iter_i){

	#if strict_iter_i_macro // Always on if MULTIP or NMIX are on. 

	Scalar u_i_new MIX(=mix_neutral(mix_is_min)); MULTIP(Int uq_i_new MIX(=0);)
	if(active) {
		NOMIX(Scalar & u_i_mix = u_i_new; MULTIP(Int & uq_i_mix = uq_i_new;)
			FLOW(Scalar * const flow_weights_mix = flow_weights; 
				 NSYM(Int * const active_side_mix = active_side;)) )
		MIX(Scalar u_i_mix; MULTIP(Int uq_i_mix;) 
			FLOW(Scalar flow_weights_mix[nact]; NSYM(Int active_side_mix[nsym];)) )

		for(Int kmix=0; kmix<nmix; ++kmix){
			const Int s = kmix*ntot;
			HFMUpdate(
				rhs, weights+kmix*nact,
				v_o+s MULTIP(,vq_o+s), v_i+s,
				ORDER2(v2_o+s MULTIP(,vq2_o+s), v2_i+s,)
				u_i MULTIP(,uq_i),
				u_i_mix MULTIP(,uq_i_mix) 
				FLOW(, flow_weights_mix NSYM(, active_side_mix))
				);
			MIX(if(mix_is_min==Greater(u_i_new MULTIP(,uq_i_new), u_i_mix MULTIP(,uq_i_mix) ) ){
				u_i_new=u_i_mix; MULTIP(uq_i_new=uq_i_mix;)
				FLOW(kmix_=kmix; 
					for(Int k=0; k<nact; ++k){flow_weights[k]=flow_weights_mix[k];}
					NSYM(for(Int k=0; k<nsym; ++k){active_side[k]=active_side_mix[k];}))
			}) // Mix and better update value
		}
	}
	__syncthreads();
	if(active DECREASING(&& Greater(u_i[n_i] MULTIP(,uq_i[n_i]),
									u_i_new  MULTIP(,uq_i_new)))) {
		u_i[n_i]=u_i_new; MULTIP(uq_i[n_i] = uq_i_new;)}
	__syncthreads();

	#else // Without strict_iter_i
	MIX("strict_iter_i is needed with mix")
	MULTIP("strict_iter_i is needed with multip")

	if(active) {
		Scalar u_i_new; MULTIP(Int uq_i_new;)
		HFMUpdate(
			rhs, weights,
			v_o MULTIP(,vq_o), v_i,
			ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
			u_i MULTIP(,uq_i),
			u_i_new MULTIP(,uq_i_new) 
			FLOW(, flow_weights NSYM(, active_side))
			);
		if(true DECREASING(&& Greater(u_i[n_i] MULTIP(,uq_i[n_i]),
									  u_i_new  MULTIP(,uq_i_new))) ) {
			u_i[n_i]=u_i_new; MULTIP(uq_i[n_i] = uq_i_new;)}
	}
	__syncthreads();

	#endif // strict_iter_i

	} // for 0<=iter_i<niter_i

	#else // nmix_adaptive_macro, 
	// Adaptive choice of kmix based on the DECREASING values assumption.
	FLOW("nmix_adaptive_macro should be deactivated with flow computation")

	Scalar u_i_new=u_i[n_i]; MULTIP(Int uq_i_new=uq_i[n_i];)
	// Variables used only if mix_is_min == true (minimum of a family of schemes)
	Int k_min=nmix/2; 
	// Variables used only if mix_is_min == false (maximum of a family of schemes)
	Scalar u_max[nmix]; MULTIP(Int uq_max[nmix];) Int order_max[nmix];

	for(Int iter_i=0; iter_i<niter_i; ++iter_i){
		if(active){
		// Select the parameter used for the next update
		Int kmix;
		if(mix_is_min){
			/* Use an equidistributed sequence of updates if iter is even,
			or the best previously seen update otherwise.*/
			if(iter_i%2==0){ kmix = eqseq<nmix>[(iter_i/2)%nmix];}
			else {           kmix = k_min;}
		} else {
			/* Try all possibilities first, then use what gave the largest result */
			if(iter_i<nmix){ kmix = iter_i;}
			else {           kmix = order_max[nmix-1];}
		}

		Scalar u_i_mix; MULTIP(Int uq_i_mix;) 
		const Int s = kmix*ntot;
		HFMUpdate(
			rhs, weights+kmix*nact,
			v_o+s MULTIP(,vq_o+s), v_i+s,
			ORDER2(v2_o+s MULTIP(,vq2_o+s), v2_i+s,)
			u_i MULTIP(,uq_i),
			u_i_mix MULTIP(,uq_i_mix) 
			);

		if(mix_is_min){ // Min of schemes.
			// Anything smaller than previously computed values is good to take
			if(Greater(u_i_new MULTIP(,uq_i_new), u_i_mix MULTIP(,uq_i_mix) ) ){
				u_i_new=u_i_mix; MULTIP(uq_i_new=uq_i_mix;)
				k_min = kmix;
			} 
		} else { // Max of schemes.
			if(iter_i<nmix){ 
				u_max[iter_i] = u_i_mix; MULTIP(uq_min[iter_i] = uq_i_mix;)
				if(iter_i==nmix-1){ // Sort the values on the last time
					#if multip_macro // Sorting methods do not accept multiprecision
					Int uq_ref=Int_Max;
					for(Int i=0; i<nmix; ++i){uq_ref = min(uq_ref,uq_max[i]);}
					Scalar u_max_abs[nmix];
					for(Int i=0; i<nmix; ++i){
						u_max_abs[i] = u_max[i]+(uq_max[i]-uq_ref)*multip_step;}
					NetworkSort::sort<nmix>(u_max_abs,order_max);
					#else
					NetworkSort::sort<nmix>(u_max,order_max);
					#endif
				}
			} else { 
				/* Insert the new value in the sorted list, and take the new top value 
				for update*/
				u_max[kmix]=u_i_mix; MULTIP(uq_max[kmix]=uq_i_mix;)
				for(Int i=nmix-2; i>=0; --i){
					const Int j = order_max[i];
					if(Greater(u_i_mix, MULTIP(uq_i_mix,) u_max[j] MULTIP(,uq_max[j]))){
						order_max[i+1]=kmix; break;
					} else { 
						order_max[i+1]=j;
					}
				const Int j=order_max[nmix-1];
				u_i_new = u_max[j]; MIX(uq_i_new = uq_max[j];)
				}
			} // Min/Max of schemes
		} // if active
		__syncthreads()
		u_i[n_i]=u_i_new; MULTIP(uq_i[n_i] = uq_i_new;)}
		__syncthreads();

	}



	#endif

}
