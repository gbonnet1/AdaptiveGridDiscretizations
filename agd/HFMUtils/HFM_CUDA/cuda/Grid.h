#pragma once
/** This file implements common facilities for bounds checking and array access.*/

bool InRange(Int x[ndim], const Int shape_[ndim]){
	for(int k=0; k<ndim; ++k){
		if(x[k]<0 || x[k]>=shape_[k]){
			return false;
		}
	}
	return true;
}

Int Index(Int x[ndim], const Int shape_i[ndim], const Int shape_o[ndim]){
	// Get the index of a point in the full array.
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

Int Index(Int x[ndim], const Int shape_[ndim]){
	Int n=0; 
	for(Int k=0; k<ndim; ++k){
		if(k>0) {n*=shape_[k];}
		n+=x[k];
	}
	return n;
}

bool GetBool(const BoolPack * arr, const Int n){
	const Int m = 8*sizeof(BoolPack);
	const Int q = n/m, r=n%m;
	return (arr[q] >> r) & BoolPack(1);
}


/** Reduction operation over a block.
Involves n_i, size_i, log2_size_i (thread identifier, number of threads, upper bound on log)
cmd should look like 
s[n_i] += s[m_i]
and the sum will be stored in stored in s[0].
CAUTION : no syncing in the last iteration, so s[0] is only visible to thread 0.
*/
#define Reduce_i(cmds) { \
	Int shift_=1; \
	for(Int k_=0; k_<log2_size_i; ++k_){ \
		const Int old_shift_=shift_; \
		shift_=shift_<<1; \
		if( (n_i%shift_)==0 ){ \
			const Int m_i = n_i+old_shift_; \
			if(m_i<size_i){ \
				cmds \
			} \
		} \
		if(k_<log2_size_i-1) {__syncthreads();} \
	} \
} \

/*
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
*/
