#include "Grid.h"
#include "HFM.h"

extern "C" {

__global__ void Update(
	Scalar * u, MULTIP(Int * uq,)
	const Scalar * geom, DRIFT(const Scalar * drift,) const BoolPack * seeds, 
	const Int * updateList_o, BoolAtom * updateNext_o){ // Used as simple booleans

//	__shared__ Int shape_o[ndim];
	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
		n_o = updateList_o[blockIdx.x];
		Grid::Position(n_o,shape_o,x_o);
	}

	__syncthreads(); // __shared__ x_o, n_o
	if(n_o==-1) return;

	Int x_i[ndim], x[ndim];
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[ndim-1]=threadIdx.z;
	for(int k=0; k<ndim; ++k){
		x[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_i = Grid::Index(x_i,shape_i);
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

			for(int l=0; l<ndim; ++l){
				y[l]   = x[l]   + eps*e[l]; 
				y_i[l] = x_i[l] + eps*e[l];
			}

			if(Grid::InRange(y_i,shape_i))  {
				v_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v_o[kv] = fact[s];)
			} else {
				v_i[kv] = -1;
				if(Grid::InRange_tot(y)) {
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

			if(Grid::InRange(y_i,shape_i))  {
				v2_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v2_o[kv] = fact2[s];)
			} else {
				v2_i[kv] = -1;
				if(Grid::InRange_tot(y)) {
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

	// Compute and save the values
	HFMIter(!isSeed, n_i, weights,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) );
	u[n] = u_i[n_i];
	MULTIP(uq[n] = uq_i[n_i];)
	
	// Find the smallest value which was changed.
	const Scalar u_diff = abs(u_old - u_i[n_i] MULTIP( + (uq_old - uq_i[n_i]) * multip_step ) );
	if( !(u_diff>tol) ){// Equivalent to u_diff<=tol, but Ignores NaNs 
		u_i[n_i]=infinity();
	} else {
		MULTIP(u_i[n_i] += uq_i[n_i]*multip_step;) // Extended accuracy ditched from this point
	}
	__syncthreads(); // Get all values before reduction

	REDUCE_i( u_i[n_i] = min(u_i[n_i],u_i[m_i]); )
	__syncthreads();  // Make u_i[0] accessible to all 

	// Tag neighbor blocks, and this particular block, for update
	if(u_i[0]!=infinity() && n_i<=2*ndim){ 
		Int k = n_i/2;
		const Int s = n_i%2;
		Int eps = 2*s-1;
		if(n_i==2*ndim){k=0; eps=0;}

		Int neigh_o[ndim];
		for(Int l=0; l<ndim; ++l) {neigh_o[l]=x_o[l];}
		neigh_o[k]+=eps;
		if(Grid::InRange(neigh_o,shape_o)) {
			updateNext_o[Grid::Index(neigh_o,shape_o)]=1;}
	}

	if(debug_print && n==0){
		printf("shape %i,%i\n",shape_tot[0],shape_tot[1]);

	}
}

} // Extern "C"