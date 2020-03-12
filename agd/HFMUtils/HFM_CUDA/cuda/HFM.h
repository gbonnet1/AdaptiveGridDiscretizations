/**
This file implements common rountines for HFM-type fast marching methods 
implemented on the GPU using CUDA.
*/

Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}

/// Ceil of the division of positive numbers
Int ceil_div(Int num, Int den){return (num+den-1)/den;}

// ---------------- Grid related methods -----------------

struct GridType {
	/** A multi-dimensional array, with data organized in blocks for faster access.
	Out of domain values yield +infinity.*/

	Int shape[ndim]; // Shape of full array
	Int shape_o[ndim]; // shape/blockShape

	bool InRange(Int x[ndim]) const {
		for(Int k=0; k<ndim; ++k){
			if(x[k]<0 || x[k]>=shape[k]){
				return false;
			}
		}
		return true;
	}

	bool InRange_i(Int x_i[ndim]) const {
		for(Int k=0; k<ndim; ++k){
			if(x_i[k]<0 || x_i[k]>=shape_i[k]){
				return false;
			}
		}
		return true;
	}

	Int Index(Int x[ndim]) const {
		// Get the index of a point in the array.
		// No bounds check 
		Int n_o=0,n_i=0;
		for(Int k=0; k<ndim; ++k){
			const Int 
			s_i = shape_i[k],
			x_o= x[k]/s_i,
			x_i= x[k]%s_i;
			if(k>0) {n_o*=shape_o[k]; n_i*=s_i;}
			n_o+=x_o; n_i+=x_i; 
		}

		const Int n=n_o*size_i+n_i;
		return n;
	}

	Int Index_i(Int x_i[ndim]) const {
		Int n_i=0; 
		for(Int k=0; k<ndim; ++k){
			if(k>0) {n_i*=shape_i[k];}
			n_i+=x_i[k];
		}
		return n_i;
	}

};

bool GetBool(const BoolPack * arr, Int n){
	const Int m = 8*sizeof(BoolPack);
	const Int q = n/m, r=n%m;
	return (arr[q] >> r) & BoolPack(1);
}

bool order(Scalar * v, Int i){
	// swaps v[i] and v[i+1] if v[i]>v[i+1]. Used in bubble sort.
	if(v[i]<=v[i+1]) return false;
	Scalar w = v[i];
	v[i] = v[i+1];
	v[i+1] = w;
	return true;
}

void bubble_sort(Scalar v[nact]){
	for(Int k=nact-1; k>=1; --k){
		for(Int r=0; r<k; ++r){
			order(v,r);
		}
	}
}


void Min0(const Int n_i, Scalar u_i[size_i]){
	/// Stores the minimum value in the first array entry
	Int shift=1;
	for(Int k=0; k<log2_size_i; ++k){
		const Int old_shift=shift;
		shift=shift<<1;
		if( (n_i%shift)==0 ){
			const Int m_i = n_i+old_shift;
			if(m_i<size_i){
				u_i[n_i] = min(u_i[n_i],u_i[m_i]);
			}
		}
		if(k<log2_size_i-1) {__syncthreads();}
	}
}


/// --------- Eulerian fast marching update operator -----------
Scalar HFMUpdate(const Int n_i, const Scalar cost,
	const Scalar v_o[ntot], const Int v_i[ntot], const Scalar u_i[size_i]){

	// Get the value for the symmetric offsets 
	// (minimal value among the right and left neighbors)
	Scalar v[nact];
	for(Int k=0; k<nsym; ++k){
		for(Int s=0; s<=1; ++s){
			const Int ks = 2*k+s;
			const Int w_i = v_i[ks];
			const Scalar v_ = w_i>=0 ? u_i[w_i] : v_o[ks];
			v[k] = s==0 ? v_ : min(v_,v[k]);
			
			/*
			if(debug_print && n_i==n_print2){
				printf("k%i,s%i, wi %i, v_ %f, v[k] %f\n",k,s,w_i,v_,v[k]);
				printf("u_i[2] %f, u_i[3] %f,  u_i[4] %f\n", u_i[2],u_i[3],u_i[4]);

			}*/
		}
	}

	// Get the value for the forward offsets
	for(Int k=0; k<nfwd; ++k){
		const Int w_i = v_i[2*nsym+k];
		v[nsym+k] = w_i>=0 ? u_i[w_i] : v_o[2*nsym+k];
	}

	bubble_sort(v);

/*
	if(debug_print && n_i==n_print2){
		printf("\n");
		for(Int k=0;k<nsym;++k){
			printf("v[%i]=%f\n",k,v[k]);
		}
		printf("value %f\n",u_i[n_i]);
	}
*/
	// Compute the update
	const Scalar vmin = v[0];
	if(vmin==infinity()){return vmin;}
	Scalar value = cost;
	Scalar a=1., b=0., c = -cost*cost;
	for(Int k=1; k<nact; ++k){
		const Scalar t = v[k] - vmin;
		if(value<=t){
//			if(debug_print && n_i==n_print2) printf("value sent %f\n\n",vmin+value); 
			return vmin+value;}
		a+=1.;
		b+=t;
		c+=t*t;
		const Scalar delta = b*b-a*c;
		const Scalar sdelta = sqrt(delta);
		value = (b+sdelta)/a;
	}
/*
	if(debug_print && n_i==n_print2){
		printf("value solved %f\n\n",vmin+value);
	}
*/
	return vmin+value;
}

void HFMIter(const bool active, const Int n_i, const Scalar cost,
	const Scalar v_o[ntot], const Int v_i[ntot], Scalar u_i[size_i]){

	#if strict_iter_i
	__shared__ Scalar u_i_new[size_i];
	for(int i=0; i<niter_i; ++i){
		if(active) {u_i_new[n_i] = HFMUpdate(n_i,cost,v_o,v_i,u_i);}
		__syncthreads();
		u_i[n_i]=u_i_new[n_i];
		__syncthreads();
	}

	#else
	for(int i=0; i<niter_i; ++i){
		if(active) {u_i[n_i] = HFMUpdate(n_i,cost,v_o,v_i,u_i);}
		__syncthreads();
	}

	#endif
}
