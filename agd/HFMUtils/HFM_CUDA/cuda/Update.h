#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "Constants.h"
#include "Grid.h"
#include "HFM.h"
#include "REDUCE_i.h"
#include "GetBool.h"
#include "Propagation.h"

/* Array suffix convention : 
 arr_t : shape_tot (Total domain shape)
 arr_o : shape_o (Grid shape)
 arr_i : shape_i (Block shape)
 arr : thread level object
*/
extern "C" {

__global__ void Update(
	Scalar * u_t, MULTIP(Int * uq_t,) STRICT_ITER_O(Scalar * uNext_t, MULTIP(Int * uqNext_t,) )
	const Scalar * geom_t, DRIFT(const Scalar * drift_t,) 
	const BoolPack * seeds_t, const Scalar * rhs_t, WALLS(wallDist_t,)
	MINCHG_FREEZE(const Scalar * minChgPrev_o, Scalar * minChgNext_o,)
	FLOW_WEIGHTS(Scalar * flow_weights_t,) FLOW_WEIGHTSUM(Scalar * flow_weightsum_t,)
	FLOW_OFFSETS(char * flow_offsets_t,) FLOW_INDICES(Int* flow_indices_t,) 
	FLOW_VECTOR(Scalar * flow_vector_t,) 
	Int * updateList_o, PRUNING(BoolAtom * updatePrev_o,) BoolAtom * updateNext_o 
	){ 

	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if( Propagation::Abort(
		updateList_o,PRUNING(updatePrev_o,) 
		MINCHG_FREEZE(minChgPrev_o,minChgNext_o,updateNext_o,)
		x_o,&n_o) ){return;} // Also sets x_o, n_o

	const Int n_i = threadIdx.x;
	Int x_i[ndim];
	Grid::Position(n_i,shape_i,x_i);

	Int x_t[ndim];
	for(int k=0; k<ndim; ++k){x_t[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_t = n_o*size_i + n_i;
	const bool isSeed = GetBool(seeds_t,n_t);
	const Scalar rhs = rhs_t[n_t];

	#ifndef isotropic_macro // No need to build the scheme in the isotropic case
	Scalar geom[geom_size];
	for(Int k=0; k<geom_size; ++k){geom[k] = geom_t[n_t+size_tot*k];}
	ADAPTIVE_WEIGHTS(Scalar weights[nactx];)
	ADAPTIVE_OFFSETS(Int offsets[nactx][ndim];)
	MIX(const bool mix_is_min = )
	scheme(geom, CURVATURE(x_t,) weights, offsets);
	#endif

	DRIFT(
	Scalar drift[ndim];
	for(Int k=0; k<ndim; ++k){drift[k] = drift_t[n_t+size_tot*k];}
	)

	const Scalar u_old = u_t[n_t]; 
	MULTIP(const Int uq_old = uq_t[n_t];)

	__shared__ Scalar u_i[size_i]; // Shared block values
	u_i[n_i] = u_old;
	MULTIP(__shared__ Int uq_i[size_i];
	uq_i[n_i] = uq_old;)

	// Apply boundary conditions
	if(isSeed){u_i[n_i]=rhs; MULTIP(uq_i[n_i]=0; Normalize(&u_i[n_i],&uq_i[n_i]);)}

	WALLS(
	__shared__ WallsT wallDist_i[size_i];
	wallDist_i[n_i] = wallDist_t[n_t];
	__syncthreads();
	)

/*	if(debug_print && n_i==0 && n_o==size_o-1){
//		printf("shape %i,%i\n",shape_tot[0],shape_tot[1]);
//		for(int k=0; k<size_i; ++k){printf("%f ",u_i[k]);}
		printf("ntotx %i, ntot %i, nsym %i, nactx_ %i, geom_size %\n",ntotx,ntot,nsym,nactx_);
		for(int k=0; k<nactx; ++k){
			printf("k=%i, offset=(%i,%i), weight=%f\n",k,offsets[k][0],offsets[k][1],weights[k]);
		}
		printf("geom : %f,%f,%f\n",geom_[0],geom_[1],geom_[2]);
		printf("size_tot : %i\n",size_tot);
		printf("scal : %f\n",scal_vmv(offsets[1],geom_,offsets[2]) );
	}*/



	FACTOR(
	Scalar x_rel[ndim]; // Relative position wrt the seed.
	const bool factors = factor_rel(x,x_rel);
	)

	// Get the neighbor values, or their indices if interior to the block
	Int    v_i[ntotx]; // Index of neighbor, if in the block
	Scalar v_o[ntotx]; // Value of neighbor, if outside the block
	MULTIP(Int vq_o[ntotx];)
	ORDER2(
		Int v2_i[ntotx];
		Scalar v2_o[ntotx];
		MULTIP(Int vq2_o[ntotx];)
		)
	Int koff=0,kv=0; 
	for(Int kmix=0; kmix<nmix; ++kmix){
	for(Int kact=0; kact<nact; ++kact){
		const Int * e = offsets[koff]; // e[ndim]
		++koff;
		SHIFT(
			Scalar fact[2]; ORDER2(Scalar fact2[2];)
			FACTOR( factor_sym(x_rel,e,fact ORDER2(,fact2)) );
			NOFACTOR( for(Int l=0; l<2; ++l){fact[l]=0; ORDER2(fact2[l]=0;)} )
			DRIFT( const Scalar s = scal_vv(drift,e); fact[0] +=s; fact[1]-=s; )
			)

		for(Int s=0; s<2; ++s){
			if(s==0 && kact>=nsym) continue;
			Int offset[ndim];
			const Int eps=2*s-1; // direction of offset
			mul_kv(eps,e,offset);

			WALLS(
			const bool visible = Visible(offset, x_t,wallDist_t, x_i,wallDist_i,n_i);
			if(!visible){
				v_i[kv]=-1; ORDER2(v2_i[kv]=-1;)
				v_o[kv]=infinity(); ORDER2(v2_o[kv]=infinity();)
				MULTIP(vq_o[kv]=0;  ORDER2(vq2_o[kv]=0;) )
				continue;}
			)

			Int y_t[ndim], y_i[ndim]; // Position of neighbor. 
			add_vv(offset,x_t,y_t);
			add_vv(offset,x_i,y_i);

			if(Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)) )  {
				v_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v_o[kv] = fact[s];)
			} else {
				v_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot) ) {
					const Int ny_t = Grid::Index_tot(y_t);
					v_o[kv] = u_t[ny_t] SHIFT(+fact[s]);
					MULTIP(vq_o[kv] = uq_t[ny_t];)
				} else {
					v_o[kv] = infinity();
					MULTIP(vq_o[kv] = 0;)
				}
			}

			ORDER2(
			add_vV(offset,y_t);
			add_vV(offset,y_i);

			if(Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)) ) {
				v2_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v2_o[kv] = fact2[s];)
			} else {
				v2_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot) ) {
					const Int ny_t = Grid::Index_tot(y_t);
					v2_o[kv] = u_t[ny_t] SHIFT(+fact2[s]);
					MULTIP(vq2_o[kv] = uq_t[ny_t];)
				} else {
					v2_o[kv] = infinity();
					MULTIP(vq2_o[kv] = 0;)
				}
			}
			) // ORDER2

			++kv;
		} // for s 
	} // for kact
	} // for kmix

	
	__syncthreads(); // __shared__ u_i
/*
	if(debug_print && n_i==17 && n_o==3){
		printf("hi there, before HFM\n");
		printf("v_o %f,%f,%f, v_i %i,%i,%i,\n",v_o[0],v_o[1],v_o[2]);
		DRIFT(printf("drift %f,%f\n", drift[0],drift[1]);)
//		printf("geom %f,%f,%f\n",geom[0],geom[1],geom[2]);
		printf("weights %f,%f,%f\n",weights[0],weights[1],weights[2]);
		printf("isSeed %i\n",isSeed);
	}*/

	FLOW(
	Scalar flow_weights[nact]; 
	Int active_side[nsym];
	Int kmix=0; 
	) 

	// Compute and save the values
	HFMIter(!isSeed, n_i, 
		rhs, MIX(mix_is_min,) weights,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) 
		FLOW(, flow_weights, active_side MIX(, &kmix) ) );

	#if strict_iter_o_macro
	uNext_t[n_t] = u_i[n_i];
	MULTIP(uqNext_t[n_t] = uq_i[n_i];)
	#else
	u_t[n_t] = u_i[n_i];
	MULTIP(uq_t[n_t] = uq_i[n_i];)
	#endif

	FLOW( // Extract and export the geodesic flow
	if(debug_print && n_i==0){
		printf("Hello, world !");
		printf("flow_weights %f,%f\n",flow_weights[0],flow_weights[1]);
		printf("active_side %i,%i\n",active_side[0],active_side[1]);
			}
	if(isSeed){ // HFM leaves these fields to their (unspecified) initial state
		for(Int k=0; k<nact; ++k){
			flow_weights[k]=0; 
			active_side[k]=0;}
		MIX(kmix=0;)
	}

	FLOW_VECTOR(Scalar flow_vector[ndim]; fill_kV(Scalar(0),flow_vector);)
	FLOW_WEIGHTSUM(Scalar flow_weightsum=0;)

	for(Int k=0; k<nact; ++k){
		FLOW_WEIGHTS(flow_weights_t[n_t+size_tot*k]=flow_weights[k];)
		FLOW_WEIGHTSUM(flow_weightsum+=flow_weights[k];)
		Int offset[ndim]; FLOW_INDICES(Int y_t[ndim];)
		const Int eps = 2*active_side[k]-1;
		for(Int l=0; l<ndim; ++l){
			offset[l] = eps*offsets[kmix*nact+k][l];
			FLOW_INDICES(y_t[l] = x_t[l]+offset[l];)
			FLOW_OFFSETS(flow_offsets_t[n_t+size_tot*(k+nact*l)]=offset[l];)
			FLOW_VECTOR(flow_vector[l]+=flow_weights[k]*offset[l];)
		}
		FLOW_INDICES(flow_indices_t[n_t+size_tot*k]=Grid::Index_tot(y_t);) 
	}
	FLOW_WEIGHTSUM(flow_weightsum_t[n_t]=flow_weightsum;)
	FLOW_VECTOR(for(Int l=0; l<ndim; ++l){flow_vector_t[n_t+size_tot*l]=flow_vector[l];})
	) // FLOW 
	
	// Find the smallest value which was changed.
	const Scalar u_diff = abs(u_old - u_i[n_i] MULTIP( + (uq_old - uq_i[n_i]) * multip_step ) );
	if( !(u_diff>tol) ){// Equivalent to u_diff<=tol, but Ignores NaNs 
		u_i[n_i]=infinity();
	} else {
		MULTIP(u_i[n_i] += uq_i[n_i]*multip_step;) // Extended accuracy ditched from this point
	}
	__syncthreads(); // Get all values before reduction

	Propagation::Finalize(
		u_i, PRUNING(updateList_o,) 
		MINCHG_FREEZE(minChgPrev_o, minChgNext_o, updatePrev_o,) updateNext_o,  
		x_o, n_o);
}

} // Extern "C"