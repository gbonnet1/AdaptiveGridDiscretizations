// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/**
This file implements the functions required to run an adaptive gauss-siedel solver.

*/

#include "Grid.h"
#include "REDUCE_i.h"

MINCHG_FREEZE(
__constant__ Scalar minChgPrev_thres, minChgNext_thres; // Previous and next threshold for freezing
)

namespace Propagation {
// Tag the neighbors for update
void TagNeighborsForUpdate(const Int n_i, const Int x_o[ndim], BoolAtom * updateNext_o){
	if(n_i>2*ndim) return;

	Int k = n_i/2;
	const Int s = n_i%2;
	Int eps = 2*s-1;
	if(n_i==2*ndim){k=0; eps=0;}

	Int neigh_o[ndim];
	for(Int l=0; l<ndim; ++l) {neigh_o[l]=x_o[l];}
	neigh_o[k]+=eps;
/*	if(debug_print && n_i==2*ndim){printf("In TagNeighbors n_i=%i, x_o=%i,%i, neigh_o=%i,%i",
		n_i,x_o[0],x_o[1],neigh_o[0],neigh_o[1]);}*/
	if(Grid::InRange_per(neigh_o,shape_o)) {
		updateNext_o[Grid::Index_per(neigh_o,shape_o)]=1 PRUNING(+n_i);}
}

bool Abort(Int * __restrict__ updateList_o, PRUNING(BoolAtom * __restrict__ updatePrev_o,) 
MINCHG_FREEZE(const Scalar * __restrict__ minChgPrev_o, Scalar * __restrict__ minChgNext_o,
 BoolAtom * __restrict__ updateNext_o,) 
Int x_o[ndim], Int & n_o){

	const Int n_i = threadIdx.x;

	PRUNING(      const Int n_o_remove = -1;)
	MINCHG_FREEZE(const Int n_o_stayfrozen = -2;
	              const Int n_o_unfreeze = -3;)

	if(n_i==0){
		n_o = updateList_o[blockIdx.x];
		MINCHG_FREEZE(const bool frozen=n_o>=size_o; if(frozen){n_o-=size_o;})
		Grid::Position(n_o,shape_o,x_o);

	#if pruning_macro
		while(true){
		const Int ks = blockIdx.x % (2*ndim+1);
	#if minChg_freeze_macro
		if(frozen){// Previously frozen block
			if(ks!=0 // Not responsible for propagation work
			|| updatePrev_o[n_o]!=0 // Someone else is working on the block
			){n_o=n_o_remove; break;} 

			const Scalar minChgPrev = minChgPrev_o[n_o];
			minChgNext_o[n_o] = minChgPrev;
			if(minChgPrev < minChgNext_thres){ // Unfreeze : tag neighbors for next update. 
				updateList_o[blockIdx.x] = n_o; n_o=n_o_unfreeze;
			} else { // Stay frozen 
				updateList_o[blockIdx.x] = n_o+size_o; n_o=n_o_stayfrozen;
			}
			break;
		}
	#endif
		// Non frozen case
		// Get the position of the block to be updated
		if(ks!=2*ndim){
			const Int k = ks/2, s = ks%2;
			x_o[k]+=2*s-1;
			PERIODIC(if(periodic_axes[k]){x_o[k] = (x_o[k]+shape_o[k])%shape_o[k];})
			// Check that the block is in range
			if(Grid::InRange(x_o,shape_o)) {n_o=Grid::Index(x_o,shape_o);}
			else {n_o=n_o_remove; break;}
		}

		// Avoid multiple updates of the same block
		if((ks+1) != updatePrev_o[n_o]) {n_o=n_o_remove; break;}
		break;
		} // while(true)
		if(n_o==n_o_remove){updateList_o[blockIdx.x]=n_o_remove;}
	#endif
	}

	__syncthreads();

	PRUNING(if(n_o==n_o_remove MINCHG_FREEZE(|| n_o==n_o_stayfrozen)) {return true;})

	MINCHG_FREEZE(
	if(n_o==n_o_unfreeze){TagNeighborsForUpdate(n_i,x_o,updateNext_o); return true;}
	if(n_i==0){updatePrev_o[n_o]=0;} // Cleanup required for MINCHG
	)

	return false;
}
	
void Finalize(
	Scalar chg_i[size_i], PRUNING(Int * __restrict__ updateList_o,) 
	FIM(const BoolAtom * __restrict__ scorePrev_o, BoolAtom * __restrict__ scoreNext_o,)
	MINCHG_FREEZE(const Scalar * __restrict__ minChgPrev_o, Scalar * __restrict__ minChgNext_o, 
	const BoolAtom * __restrict__ updatePrev_o,) BoolAtom * __restrict__ updateNext_o,  
	Int x_o[ndim], Int n_o
	){
	const Int n_i = threadIdx.x;

	MINCHG_FREEZE(__shared__ Scalar minChgPrev; if(n_i==0){minChgPrev = minChgPrev_o[n_o];})
	REDUCE_i( chg_i[n_i] = min(chg_i[n_i],chg_i[m_i]); )

	__syncthreads();  // Make u_i[0] accessible to all, also minChgPrev
	Scalar minChg = chg_i[0];
	
	// Tag neighbor blocks, and this particular block, for update

#if minChg_freeze_macro // Propagate if change is small enough (w.r.t global threshold)
	const bool frozenPrev = minChgPrev>=minChgPrev_thres && minChgPrev!=infinity();
	if(frozenPrev){minChg = min(minChg,minChgPrev);}
	const bool propagate = minChg < minChgNext_thres;
	const bool freeze = !propagate && minChg!=infinity();
	if(n_i==size_i-2) {minChgNext_o[n_o] = minChg;}

#elif fim_macro // FIM : Propagate if block has converged or is close to the front
	const BoolAtom scoreConverged = 1;
	bool propagate = false;
	if(minChg==infinity()) { // Block converged. 
		scoreNext_o[n_o]=scoreConverged;
		if(scorePrev_o[n_o]!=scoreConverged){
			propagate = true;}
	} else { // Block non-converged. Tag for next update.
		updateNext_o[n_o]=1;
		if(scoreConverged>1){ // Variant of FIM allowing wider fronts
			// Get the maximal score among active neigbors. Substract one.
			if(n_i<2*ndim){
				// TODO. Check neighbors, etc
			}
		} else {
			scoreNext_o[n_o] = 0;
		}
	}

#else // AGSI : Propagate as soon as something changed
	const bool propagate = minChg != infinity();
#endif

	if(propagate){TagNeighborsForUpdate(n_i,x_o,updateNext_o);}
	PRUNING(if(n_i==size_i-1){updateList_o[blockIdx.x] 
		= propagate ? n_o : MINCHG_FREEZE(freeze ? (n_o+size_o) :) -1;})
}

}