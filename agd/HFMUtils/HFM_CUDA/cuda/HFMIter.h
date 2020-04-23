#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0
/**
This file implements the of a block of values, in the HFM algorithm.
*/

#include "HFM.h"

void HFMIter(const bool active, const Int n_i, 
	const Scalar rhs, MIX(const bool mix_is_min,) const Scalar weights[__restrict__ nactx],
	const Scalar v_o[__restrict__ ntotx], MULTIP(const Int vq_o[__restrict__ ntotx],) 
	const Int v_i[__restrict__ ntotx], 
	ORDER2(const Scalar v2_o[__restrict__ ntotx], MULTIP(const Int vq2_o[__restrict__ ntotx],) 
		const Int v2_i[__restrict__ ntotx],)
	Scalar u_i[__restrict__ size_i] MULTIP(, Int uq_i[__restrict__ size_i]) 
	FLOW(, Scalar flow_weights[__restrict__ nact] NSYM(, Int active_side[__restrict__ nsym]) 
		MIX(, Int & kmix_) ) ){


	if(strict_iter_i_macro || nmix>1){
	for(int i=0; i<niter_i; ++i){
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
					n_i, rhs, weights+kmix*nact,
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
	} // for niter_i

	} else { // without strict_iter_i
	for(int i=0; i<niter_i; ++i){
		Scalar u_i_new; MULTIP(Int uq_i_new;)
		if(active) {
			HFMUpdate(
				n_i, rhs, weights,
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
	}

	} // strict_iter_i
}
