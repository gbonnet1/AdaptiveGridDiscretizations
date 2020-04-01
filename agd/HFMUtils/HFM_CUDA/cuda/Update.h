// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include "Grid.h"
#include "HFM.h"

MINCHG_FREEZE(
__constant__ Scalar minChgPrev_thres, minChgNext_thres; // Previous and next threshold for freezing
)

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
	if(Grid::InRange_per(neigh_o,shape_o)) {
		updateNext_o[Grid::Index_per(neigh_o,shape_o)]=1 PRUNING(+n_i);}
}


extern "C" {

__global__ void Update(
	Scalar * u, MULTIP(Int * uq,) STRICT_ITER_O(Scalar * uNext, MULTIP(Int * uqNext,) )
	const Scalar * geom, DRIFT(const Scalar * drift,) const BoolPack * seeds, 
	MINCHG_FREEZE(const Scalar * minChgPrev_o, Scalar * minChgNext_o,)
	Int * updateList_o, PRUNING(BoolAtom * updatePrev_o,) BoolAtom * updateNext_o ){ 
	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
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
			){n_o=-1; break;} 

			const Scalar minChgPrev = minChgPrev_o[n_o];
			minChgNext_o[n_o] = minChgPrev;
			if(minChgPrev < minChgNext_thres){ // Unfreeze : tag neighbors for next update. 
				updateList_o[blockIdx.x] = n_o; n_o=-3;
			} else { // Stay frozen 
				updateList_o[blockIdx.x] = n_o+size_o; n_o=-2;
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
			else {n_o=-1; break;}
		}

		// Avoid multiple updates of the same block
		if((ks+1) != updatePrev_o[n_o]) {n_o=-1; break;}
		break;
		} // while(true)
		if(n_o==-1){updateList_o[blockIdx.x]=-1;}
	#endif
	}

	__syncthreads(); // __shared__ x_o, n_o
	PRUNING(if(n_o==-1 MINCHG_FREEZE(|| n_o==-2)) {return;})

	Int x_i[ndim], x[ndim];
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[ndim-1]=threadIdx.z;
	for(int k=0; k<ndim; ++k){
		x[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_i = Grid::Index(x_i,shape_i);
	MINCHG_FREEZE(
		if(n_o==-3){TagNeighborsForUpdate(n_i,x_o,updateNext_o); return;}
		if(n_i==0){updatePrev_o[n_o]=0;} // Cleanup required for MINCHG
		)

	const Int n = n_o*size_i + n_i;
	const bool isSeed = Grid::GetBool(seeds,n);

	Scalar weights[nactx_];
	ISO(weights[0] = geom[n];) // Offsets are constant.
	ANISO(
		Scalar geom_[geom_size];
		Int offsets[nactx][ndim];
		for(Int k=0; k<geom_size; ++k){
			geom_[k] = geom[n+size_tot*k];}
		scheme(geom_, CURVATURE(x,) weights,offsets);
	)
	DRIFT(
		Scalar drift_[ndim];
		for(Int k=0; k<ndim; ++k){
			drift_[k] = drift[n+size_tot*k];}
	)

	const Scalar u_old = u[n]; 
	__shared__ Scalar u_i[size_i]; // Shared block values
	u_i[n_i] = u_old;

	MULTIP(
	const Int uq_old = uq[n];
	__shared__ Int uq_i[size_i];
	uq_i[n_i] = uq_old;
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
			DRIFT( const Scalar s = scal_vv(drift_,e); fact[0] +=s; fact[1]-=s; )
			)

		for(Int s=0; s<2; ++s){
			if(s==0 && kact>=nsym) continue;

			Int y[ndim], y_i[ndim]; // Position of neighbor. 
			const Int eps=2*s-1;

			for(Int l=0; l<ndim; ++l){
				y[l]   = x[l]   + eps*e[l]; 
				y_i[l] = x_i[l] + eps*e[l];
			}

			if(Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y,shape_tot)) )  {
				v_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v_o[kv] = fact[s];)
			} else {
				v_i[kv] = -1;
				if(APERIODIC(Grid::InRange(y,shape_tot)) 
					PERIODIC(Grid::InRange_per(y,shape_tot)) ) {
					const Int ny = Grid::Index_tot(y);
					v_o[kv] = u[ny] SHIFT(+fact[s]);
					MULTIP(vq_o[kv] = uq[ny];)
				} else {
					v_o[kv] = infinity();
					MULTIP(vq_o[kv] = 0;)
				}
			}

			ORDER2(
			for(int l=0; l<ndim; ++l){
				y[l]   +=  eps*e[l]; 
				y_i[l] +=  eps*e[l];
			}

			if(Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y,shape_tot)) ) {
				v2_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v2_o[kv] = fact2[s];)
			} else {
				v2_i[kv] = -1;
				if(APERIODIC(Grid::InRange(y,shape_tot)) 
					PERIODIC(Grid::InRange_per(y,shape_tot)) ) {
					const Int ny = Grid::Index_tot(y);
					v2_o[kv] = u[ny] SHIFT(+fact2[s]);
					MULTIP(vq2_o[kv] = uq[ny];)
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

	if(debug_print && n_i==17 && n_o==3){
		printf("hi there, before HFM\n");
		printf("v_o %f,%f,%f, v_i %i,%i,%i,\n",v_o[0],v_o[1],v_o[2]);
		DRIFT(printf("drift_ %f,%f\n", drift_[0],drift_[1]);)
		printf("geom_ %f,%f,%f\n",geom_[0],geom_[1],geom_[2]);
		printf("weights %f,%f,%f\n",weights[0],weights[1],weights[2]);
		printf("isSeed %i\n",isSeed);
	}


	// Compute and save the values
	HFMIter(!isSeed, n_i, weights,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) );

	#if strict_iter_o_macro
	uNext[n] = u_i[n_i];
	MULTIP(uqNext[n] = uq_i[n_i];)
	#else
	u[n] = u_i[n_i];
	MULTIP(uq[n] = uq_i[n_i];)
	#endif

	if(debug_print && x_i[0]==2 && x_i[1]==1 && x_o[0]==1 && x_o[1]==1){
		printf("n_i %i, n_o %o, u_i[n_i] %f\n",n_i,n_o,u_i[n_i]);
		for(Int k=0; k<size_i; ++k){printf(" %f",u_i[k]);}
		/*
		printf("shape %i,%i\n",shape_tot[0],shape_tot[1]);
		for(int k=0; k<size_i; ++k){printf("%f ",u_i[k]);}
		*/
	}

	// Find the smallest value which was changed.
	const Scalar u_diff = abs(u_old - u_i[n_i] MULTIP( + (uq_old - uq_i[n_i]) * multip_step ) );
	if( !(u_diff>tol) ){// Equivalent to u_diff<=tol, but Ignores NaNs 
		u_i[n_i]=infinity();
	} else {
		MULTIP(u_i[n_i] += uq_i[n_i]*multip_step;) // Extended accuracy ditched from this point
	}
	__syncthreads(); // Get all values before reduction

	MINCHG_FREEZE(__shared__ Scalar minChgPrev; if(n_i==0){minChgPrev = minChgPrev_o[n_o];})


	REDUCE_i( u_i[n_i] = min(u_i[n_i],u_i[m_i]); )


	__syncthreads();  // Make u_i[0] accessible to all, also minChgPrev
	Scalar minChg = u_i[0];

	if(debug_print && x_i==0){
		printf("hello world\n");
		printf("geom : %f,%f,%f\n",geom_[0],geom_[1],geom_[2]);
		DRIFT(printf("drift : %f,%f\n",drift_[0],drift_[1]);)
	}

/*
	if(debug_print && n_i==0 && n_o==0){
//		printf("shape %i,%i\n",shape_tot[0],shape_tot[1]);
//		for(int k=0; k<size_i; ++k){printf("%f ",u_i[k]);}
		printf("ntotx %i, ntot %i, nsym %i\n",ntotx,ntot,nsym);
		for(int k=0; k<nactx; ++k){
			printf("k=%i, offset=(%i,%i), weight=%f\n",k,offsets[k][0],offsets[k][1],weights[k]);
		}
		printf("geom : %f,%f,%f\n",geom_[0],geom_[1],geom_[2]);
		printf("size_tot : %i\n",size_tot);
		printf("scal : %f\n",scal_vmv(offsets[1],geom_,offsets[2]) );
	}
*/


	// Tag neighbor blocks, and this particular block, for update

#if minChg_freeze_macro // Propagate if change is small enough
	const bool frozenPrev = minChgPrev>=minChgPrev_thres && minChgPrev!=infinity();
	if(frozenPrev){minChg = min(minChg,minChgPrev);}
	const bool propagate = minChg < minChgNext_thres;
	const bool freeze = !propagate && minChg!=infinity();
	if(n_i==size_i-2) {minChgNext_o[n_o] = minChg;}
#else // Propagate as soon as something changed
	const bool propagate = minChg != infinity();
#endif

	if(propagate){TagNeighborsForUpdate(n_i,x_o,updateNext_o);}
	PRUNING(if(n_i==size_i-1){updateList_o[blockIdx.x] 
		= propagate ? n_o : MINCHG_FREEZE(freeze ? (n_o+size_o) :) -1;})
}

} // Extern "C"