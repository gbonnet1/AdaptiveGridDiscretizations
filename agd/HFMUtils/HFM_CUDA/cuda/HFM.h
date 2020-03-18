#pragma once
/**
This file implements common rountines for HFM-type fast marching methods 
running on the GPU based on CUDA.
*/

#ifndef isotropic_macro
#define isotropic_macro 0
#endif

#if isotropic_macro
const Int nact_ = 1;
#define ISO(...) __VA_ARGS__
#define ANISO(...)
#else 
const Int nact_ = nact;
#define ISO(...) 
#define ANISO(...) __VA_ARGS__
#endif

/// Normalizes a multi-precision variable so that u is as small as possible
MULTIP( 
void Normalize(Scalar * u, Int * uq){
	if( *u<multip_max ){
		const Int n = Int(*u / multip_step);
		*u -= n*multip_step;
		*uq += n;
	} 
} )

/// Compares u and v, possibly in multi-precision
bool Greater(const Scalar u MULTIP(, const Int uq), const Scalar v MULTIP(, const Int vq) ){
	NOMULTIP(return u>v;)
	MULTIP(return u-v > (vq-uq)*multip_step; )
}

// --- Gets all the neighbor values ---
void HFMNeighbors(const Int n_i, 
	const Scalar v_o[ntot],   MULTIP(const Int vq_o[ntot],) const Int v_i[ntot], 
	const Scalar u_i[size_i], MULTIP(const Int uq_i[size_i],)
	Scalar v[nact], MULTIP(Int vq[nact],) Int order[nact]){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	for(Int k=0; k<nsym; ++k){
		for(Int s=0; s<=1; ++s){
			const Int ks = 2*k+s;
			const Int w_i = v_i[ks];
			Scalar v_ MULTIP(,vq_);
			if(w_i>=0){
				v_ = u_i[w_i] FACTOR(+v_o[ks]);
				MULTIP(vq_ = uq_i[w_i];)
			} else {
				v_ = v_o[ks];
				MULTIP(vq_ = vq_o[ks];)
			}

			if(s==0) { 
				v[k] = v_; MULTIP(vq[k] = vq_;)
			} else if( Greater(v[k] MULTIP(, vq[k]), v_ MULTIP(, vq_)) ){
				v[k] = v_; MULTIP(vq[k] = vq_;)
			}
		}
	}

	// Get the value for the forward offsets
	for(Int k=0; k<nfwd; ++k){
		const Int nk = nsym+k, n2k = 2*nsym+k;
		const Int w_i = v_i[n2k];
		if(w_i>=0){
			v[nk] = u_i[w_i];
			MULTIP(vq[nk] = uq_i[w_i];)
		} else {
			v[nk] = v_o[n2k];
			MULTIP(vq[nk] = vq_o[n2k];)
		}
	}

	// Bubble sort
	for(Int k=0; k<nact; ++k) {order[k]=k;}
	for(Int k=nact-1; k>=1; --k){
		for(Int r=0; r<k; ++r){
			const Int i=order[r], j=order[r+1];
			if( Greater(v[i] MULTIP(,vq[i]), v[j] MULTIP(,vq[j]) ) ){
				// swap( order[k], order[k+1] )
				const Int s = order[r];
				order[r] = order[r+1];
				order[r+1] = s;
			}
		}
	}

} // HFMNeighbors


/// --------- Eulerian fast marching update operator -----------
void HFMUpdate(const Int n_i, const Scalar weights[nact_],
	const Scalar v_o[ntot], MULTIP(const Int vq_o[ntot],) const Int v_i[ntot], 
	const Scalar u_i[size_i], MULTIP(const Int uq_i[size_i],)
	Scalar * u_out MULTIP(,Int * uq_out) ){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	Scalar v[nact]; 
	MULTIP(Int vq[nact];)
	Int order[nact];

	HFMNeighbors(n_i,
		v_o MULTIP(,vq_o), v_i,
		u_i MULTIP(,uq_i), 
		v MULTIP(,vq), order);

	if(debug_print && n_i==1){
		printf("HFMUpdate ni : %i\n",n_i);
		printf("v : %f %f\n",v[0],v[1]);
		printf("order : %i %i\n",order[0],order[1]);
/*		printf("multip_step %f, multip_max %f\n",multip_step,multip_max);
		printf("vq : %i %i\n",vq[0],vq[1]);*/
	}


	// Compute the update
	const Int k=order[0];
	const Scalar vmin = v[k]; MULTIP(const Int vqmin = vq[k];)
	if(vmin==infinity()){*u_out = vmin; MULTIP(*uq_out=0;) return;}
	ISO(const Scalar cost = weights[0];
		Scalar value = cost;
		Scalar a=1., b=0., c = -cost*cost;)
	ANISO(const Scalar w = weights[k];
		Scalar value = 1./sqrt(w);
		Scalar a=w, b=0., c=-1.;
		)
	for(Int k_=1; k_<nact; ++k_){
		const Int k = order[k_];
		const Scalar t = (v[k] - vmin) MULTIP( + (vq[k]-vqmin)*multip_step );
		if(value<=t){break;}
		ISO(a+=1.;
			b+=t;
			c+=t*t;)
		ANISO(const Scalar w = weights[k];
			a+=w;
			b+=w*t;
			c+=w*t*t;)
		const Scalar delta = b*b-a*c;
		const Scalar sdelta = sqrt(delta);
		value = (b+sdelta)/a;
	}

/*	if(debug_print && n_i==1){
		Scalar u_out = vmin+value; Int uq_out = vqmin;
		printf("u_out %f, uq_out %i\n",u_out,uq_out);
		Normalize(&u_out,&uq_out);
		printf("u_out %f, uq_out %i\n",u_out,uq_out);
	}*/

	*u_out = vmin+value; MULTIP(*uq_out = vqmin; Normalize(u_out,uq_out); )
}

void HFMIter(const bool active, const Int n_i, const Scalar weights[nact_],
	const Scalar v_o[ntot], MULTIP(const Int vq_o[ntot],) const Int v_i[ntot], 
	Scalar u_i[size_i] MULTIP(, Int uq_i[size_i]) ){

	#if strict_iter_i
	__shared__ Scalar u_i_new[size_i];
	MULTIP(__shared__ Int uq_i_new[size_i];)
	for(int i=0; i<niter_i; ++i){
		if(active) {HFMUpdate(
			n_i,weights,
			v_o MULTIP(,vq_o), v_i,
			u_i MULTIP(,uq_i),
			&u_i_new[n_i] MULTIP(,&uq_i_new[n_i]) 
			);}
		__syncthreads();
		u_i[n_i]=u_i_new[n_i]; MULTIP(uq_i[n_i] = uq_i_new[n_i];)
		__syncthreads();
	}

	#else
	for(int i=0; i<niter_i; ++i){
		if(active) {HFMUpdate(
			n_i,weights,
			v_o MULTIP(,vq_o), v_i,
			u_i MULTIP(,uq_i),
			&u_i[n_i] MULTIP(,&uq_i[n_i]) 
		 );}
		__syncthreads();
	}

	#endif
}
