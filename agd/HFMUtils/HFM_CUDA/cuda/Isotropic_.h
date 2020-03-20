#pragma once 

#define isotropic_macro 1

#include "Constants.h"
#include "Grid.h"
#include "HFM.h"

extern "C" {

FACTOR(
/** Returns the perturbations involved in the factored fast marching method.
Input : x= relative position w.r.t the seed, e finite difference offset.*/
void factor_sym(const Scalar x[ndim], const Int e[ndim], 
	Scalar fact[2] ORDER2(,Scalar fact2[2])){
	// Compute some scalar products
	Scalar xx=0.,xe=0.,ee=0.;
	for(Int k=0; k<ndim; ++k){
		xx += x[k]*x[k];
		xe += x[k]*e[k];
		ee += e[k]*e[k];
	}
	const Scalar Nx = sqrt(xx), // |x|
	Nxme = sqrt(xx-2*xe+ee), // |x-e|
	Nxpe = sqrt(xx+2*xe+ee); // |x+e|
	const Scalar 
	Nx_Nxme = ( 2*xe - ee)/(Nx + Nxme), // |x|-|x-e| computed in a stable way
	Nx_Nxpe = (-2*xe - ee)/(Nx + Nxpe), // |x|-|x+e| computed in a stable way
	grad_e = xe/Nx; // <e, x/|x|>
	fact[0] = -grad_e + Nx_Nxme; // <-e,x/|x|> + |x|-|x-e|
	fact[1] =  grad_e + Nx_Nxpe; // < e,x/|x|> + |x|-|x+e|

	ORDER2(
	const Scalar 
	Nxme2 = sqrt(xx-4*xe+4*ee), // |x-2e| 
	Nxpe2 = sqrt(xx+4*xe+4*ee); // |x+2e| 
	const Scalar 
	Nxme2_Nxme = (-2*xe + 3*ee)/(Nxme+Nxme2), // |x-2e|-|x-e| computed in a stable way
	Nxpe2_Nxpe = ( 2*xe + 3*ee)/(Nxpe+Nxpe2); // |x+2e|-|x+e| computed in a stable way
	fact2[0] = 2*fact[0]-(Nx_Nxme + Nxme2_Nxme); // |x|-2|x-e|+|x-2e|
	fact2[1] = 2*fact[1]-(Nx_Nxpe + Nxpe2_Nxpe); // |x|-2|x+e|+|x+2e|
	)

	for(Int k=0; k<2; ++k){
		fact[k]*=factor_metric[0];
		ORDER2(fact2[k]*=factor_metric[0];)
	}	
}
) // FACTOR

__global__ void Update(
	Scalar * u, MULTIP(Int * uq,)
	const Scalar * metric, const BoolPack * seeds, 
	const Int * updateList_o, BoolAtom * updateNext_o){ // Used as simple booleans

//	__shared__ Int shape_o[ndim];
	__shared__ Int x_o[ndim];
	__shared__ Int n_o;

	if(threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0){
		n_o = updateList_o[blockIdx.x];
		Grid::Position(n_o,shape_o,x_o);
	}

	__syncthreads(); // __shared__ x_o, n_o

	Int x_i[ndim], x[ndim];
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[ndim-1]=threadIdx.z;
	for(int k=0; k<ndim; ++k){
		x[k] = x_o[k]*shape_i[k]+x_i[k];}

	const Int n_i = Grid::Index(x_i,shape_i);
	const Int n = n_o*size_i + n_i;

	const bool isSeed = Grid::GetBool(seeds,n);
	const Scalar cost = metric[n];

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
	Int    v_i[ntot]; // Index of neighbor, if in the block
	Scalar v_o[ntot]; // Value of neighbor, if outside the block
	MULTIP(Int vq_o[ntot];)
	ORDER2(
		Int v2_i[ntot];
		Scalar v2_o[ntot];
		MULTIP(Int vq2_o[ntot];)
		)
	for(Int k=0,ks=0; k<nsym; ++k){
		const Int * e = offsets[k]; // e[ndim]
		FACTOR(
			Scalar fact[2]; ORDER2(Scalar fact2[2];)
			factor_sym(x_rel,e,fact ORDER2(,fact2));
			)

		for(Int s=0; s<2; ++s){
			
			Int y[ndim], y_i[ndim]; // Position of neighbor. 
			const Int eps=2*s-1;

			for(int l=0; l<ndim; ++l){
				y[l]   = x[l]   + eps*e[l]; 
				y_i[l] = x_i[l] + eps*e[l];
			}

			if(Grid::InRange(y_i,shape_i))  {
				v_i[ks] = Grid::Index(y_i,shape_i);
				FACTOR(v_o[ks] = fact[s];)
			} else {
				v_i[ks] = -1;
				if(Grid::InRange(y,shape_tot)) {
					const Int ny = Grid::Index(y,shape_i,shape_o);
					v_o[ks] = u[ny] FACTOR(+fact[s]);
					MULTIP(vq_o[ks] = uq[ny];)
				} else {
					v_o[ks] = infinity();
					MULTIP(vq_o[ks] = 0;)
				}
			}

			ORDER2(
			for(int l=0; l<ndim; ++l){
				y[l]   +=  eps*e[l]; 
				y_i[l] +=  eps*e[l];
			}

			if(Grid::InRange(y_i,shape_i))  {
				v2_i[ks] = Grid::Index(y_i,shape_i);
				FACTOR(v2_o[ks] = fact2[s];)
			} else {
				v2_i[ks] = -1;
				if(Grid::InRange(y,shape_tot)) {
					const Int ny = Grid::Index(y,shape_i,shape_o);
					v2_o[ks] = u[ny] FACTOR(+fact2[s]);
					MULTIP(vq2_o[ks] = uq[ny];)
				} else {
					v2_o[ks] = infinity();
					MULTIP(vq2_o[ks] = 0;)
				}
			}
			) // ORDER2

			++ks;
		}
	}
	__syncthreads(); // __shared__ u_i

	// Compute and save the values
	HFMIter(!isSeed, n_i, &cost,
		v_o MULTIP(,vq_o), v_i, 
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i) );
	u[n] = u_i[n_i];
	MULTIP(uq[n] = uq_i[n_i];)

	if(debug_print && n_i==1){
//		printf("u_i %f, uq_i %i",u_i[n_i],uq_i[n_i]);
	}
	
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