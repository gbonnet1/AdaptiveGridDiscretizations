/** This file implements a 'vector decomposition' based on the decomposition of a the 
positive definite matrix obtained by relaxing the self outer product of that vector.
In particular, decomp_m must be defined.
*/

__constant__ Scalar decomp_v_relax = 0.01; // Relaxation parameter for Selling_v. 
__constant__ Scalar decomp_v_cosmin2 = 2./3.; // Relaxation parameter for Selling_v.

// Based on selling decomposition, with some relaxation, reorienting of offsets, and pruning of weights
void Selling_v(const Scalar v[ndim], Scalar weights[symdim], OffsetT offsets[symdim][ndim]){

	// Build and decompose the relaxed self outer product of v
	Scalar m[symdim];
	self_outer_relax_v(v,decomp_v_relax,m);	
	Selling_m(m,weights,offsets);
	const Scalar vv = scal_vv(v,v);

	// Redirect offsets in the direction of v, and eliminate those which deviate too much.
	for(Int k=0; k<symdim; ++k){
		OffsetT * e = offsets[k]; // e[ndim]
		const Scalar ve = scal_vv(v,e), ee = scal_vv(e,e);
		if(ve*ve < vv*ee* decomp_v_cosmin2){weights[k]=0; continue;}
		if(ve>0){neg_V(e);} // Note : we want ve<0.
	}
}