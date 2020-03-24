// Accumulate in a shared array (within a block)
void Accumulate(Int * acc_i, const Int n_i, const Int log2_size_i){
	Int mask1 = 1, mask2= -1; // mask2 = "11111...1" in binary
	for(Int k=0; k<log2_size_i; ++k){
		if(n_i & mask1){
			const Int m_i = (n_i & mask2) - 1;
			acc_i[n_i] += acc_i[m_i];
		}
		mask1 = mask1<<1; mask2 = mask2<<1;
		__syncthreads();
	} 
}