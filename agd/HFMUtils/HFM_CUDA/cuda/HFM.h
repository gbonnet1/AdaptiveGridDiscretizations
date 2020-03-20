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
	ORDER2(const Scalar v2_o[ntot],   MULTIP(const Int vq2_o[ntot],) const Int v2_i[ntot],)
	const Scalar u_i[size_i], MULTIP(const Int uq_i[size_i],)
	Scalar v[nact], MULTIP(Int vqmin[1],) ORDER2(bool order2[nact],)
	Int order[nact]){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	MULTIP(Int vq[nact];)
	ORDER2(Int side[nsym];)
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
				v[k] = v_; MULTIP(vq[k] = vq_;) ORDER2(side[k] = 0;)
			} else if( Greater(v[k] MULTIP(, vq[k]), v_ MULTIP(, vq_)) ){
				v[k] = v_; MULTIP(vq[k] = vq_;) ORDER2(side[k] = 1;)
			}
		}
	}

	// Get the value for the forward offsets
	for(Int k=0; k<nfwd; ++k){
		const Int nk = nsym+k, n2k = 2*nsym+k;
		const Int w_i = v_i[n2k];
		if(w_i>=0){
			v[nk] = u_i[w_i] FACTOR(+v_o[n2k]);
			MULTIP(vq[nk] = uq_i[w_i];)
		} else {
			v[nk] = v_o[n2k];
			MULTIP(vq[nk] = vq_o[n2k];)
		}
	}

	// Find the minimum value for the multi-precision int, and account for it
	MULTIP(
	*vqmin = Int_MAX;
	for(Int k=0; k<nact; ++k){
		if(v[k]<infinity()){
			*vqmin = min(*vqmin,vq[k]);}
	}

	for(Int k=0; k<nact; ++k){
		v[k] += (vq[k]-*vqmin)*multip_step;}
	)

	ORDER2(
	// Set the threshold for second order accuracy
	const Scalar u0 = u_i[n_i] MULTIP(+ (uq_i[n_i] - *vqmin)*multip_step);
	Scalar diff_max = 0;
	for(Int k=0; k<nact; ++k){
		diff_max=max(diff_max, u0 - v[k]);}

	for(Int k=0; k<nact; ++k){
		// Get the further neighbor value
		const Int ks = k<nsym ? (2*k+side[k]) : (k+nsym);
		const Int w_i=v2_i[ks];
		Scalar v2;
		if(w_i>=0){
			v2 = u_i[w_i] MULTIP(+ (uq_i[w_i]-*vqmin)*multip_step) FACTOR(+v2_o[ks]);
		} else {
			v2 = v2_o[ks] MULTIP(+ (vq2_o[ks]-*vqmin)*multip_step);
		}

		// Compute the second order difference, and compare
		const Scalar diff2 = abs(u0-2*v[k]+v2);
		if(diff2 < order2_threshold*diff_max){
			order2[k]=true;
			v[k] += (v[k]-v2)/3.;
		} else {
			order2[k]=false;
		}
	}
	)

	// Bubble sort the neighbor values
	for(Int k=0; k<nact; ++k) {order[k]=k;}
	for(Int k=nact-1; k>=1; --k){
		for(Int r=0; r<k; ++r){
			const Int i=order[r], j=order[r+1];
			if( v[i] > v[j] ){ 
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
	ORDER2(const Scalar v2_o[ntot], MULTIP(const Int vq2_o[ntot],) const Int v2_i[ntot],)
	const Scalar u_i[size_i], MULTIP(const Int uq_i[size_i],)
	Scalar * u_out MULTIP(,Int * uq_out) ){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	Scalar v[nact]; 
	MULTIP(Int vqmin[1];) // shared value vqmin
	ORDER2(bool order2[nact];) // Wether second order is active for this neighbor
	Int order[nact];

	HFMNeighbors(n_i,
		v_o MULTIP(,vq_o), v_i,
		ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
		u_i MULTIP(,uq_i), 
		v MULTIP(,vqmin) ORDER2(,order2),
		order);

	if(debug_print && n_i==1){
		printf("HFMUpdate ni : %i\n",n_i);
		printf("v : %f %f\n",v[0],v[1]);
		printf("order : %i %i\n",order[0],order[1]);
/*		printf("multip_step %f, multip_max %f\n",multip_step,multip_max);
		printf("vq : %i %i\n",vq[0],vq[1]);*/
	}


	// Compute the update
	const Int k=order[0];
	const Scalar vmin = v[k]; 
	if(vmin==infinity()){*u_out = vmin; MULTIP(*uq_out=0;) return;}
	
	const Scalar rhs = ISO(weights[0]) ANISO(1.);
	Scalar w = ISO(1.) ANISO(*weights[k]); 
	ORDER2(if(order2[k]) w*=9./4.;)

	Scalar value = rhs/sqrt(w);
	Scalar a=w, b=0., c=-rhs*rhs;

	for(Int k_=1; k_<nact; ++k_){
		const Int k = order[k_];
		const Scalar t = v[k] - vmin;
		if(value<=t){break;}
		Scalar w = ISO(1.) ANISO(*weights[k]); 
		ORDER2(if(order2[k]) w*=9./4.;)
		a+=w;
		b+=w*t;
		c+=w*t*t;
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

	*u_out = vmin+value; MULTIP(*uq_out = vqmin[0]; Normalize(u_out,uq_out); )
}

void HFMIter(const bool active, const Int n_i, const Scalar weights[nact_],
	const Scalar v_o[ntot], MULTIP(const Int vq_o[ntot],) const Int v_i[ntot], 
	ORDER2(const Scalar v2_o[ntot], MULTIP(const Int vq2_o[ntot],) const Int v2_i[ntot],)
	Scalar u_i[size_i] MULTIP(, Int uq_i[size_i]) ){

	#if strict_iter_i
	__shared__ Scalar u_i_new[size_i];
	MULTIP(__shared__ Int uq_i_new[size_i];)
	for(int i=0; i<niter_i; ++i){
		if(active) {HFMUpdate(
			n_i,weights,
			v_o MULTIP(,vq_o), v_i,
			ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
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
			ORDER2(v2_o MULTIP(,vq2_o), v2_i,)
			u_i MULTIP(,uq_i),
			&u_i[n_i] MULTIP(,&uq_i[n_i]) 
		 );}
		__syncthreads();
	}

	#endif
}
