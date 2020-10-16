// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file setups the finite difference scheme used in the Eulerian fast marching method.
The scheme parameters (weights, offsets,drift,mix) are called. 
The finite differences which fall in the shape_i block are identified. 
The values associated to other finite diffferences are imported.
*/


//typedef const OffsetT (*OffsetVecT)[ndim]; // OffsetVecT[][ndim]
//typedef const Scalar (*DriftVecT)[ndim]; // DriftVectT[][ndim]

/* 
Something horribly wrong happens with this code : 
- if it is included inline (using finitedifferences_inlinecode_macro) then it works fine
- if it is called as a function, then it does NOT always work fine. 
Symtom : The variable v_i[1] is modified multiple times, without reason.
Reproducibility : 
 - Happens with Elastica2 model, and import_scheme_macro = false.
 - Does not happen with Elastica2 model, and import_scheme_macro = true.
  Does not seem to happen with most other models either.

Issue could be related with local cuda device memory overuse (??), since the Elastica2 
model has many weights and offsets. However, this is insensitive to shape_i (block size), 
and to using OffsetT = int8 instead of int32.

Further observations : 
- This code is included in GeodesicODE_Opt.h, when geodesic_online_flow is activated.
 In that place, it is not possible to solve the problem the same way, because the code is
 already included in a subfunction. Problems are encountered with all curvature penalized
 models (ReedsShepp2, ... Elastica2), and again disappear when scheme precomputation is 
 activated.
*/

#if !finitedifferences_inlinecode_macro
__global__ void FiniteDifferences(
	
	// Value function (problem unknown)
	const Scalar * __restrict__ u_t, MULTIP(const Int * __restrict__ uq_t,) 
	WALLS(const WallT * __restrict__ wallDist_t, const WallT wallDist_i[__restrict__ size_i],)

	// Structure of the finite differences (input, output)
	const OffsetT offsets[__restrict__ nactx][ndim], DRIFT(const Scalar drift[__restrict__ nmix][ndim],)
	Int v_i[__restrict__ ntotx], Scalar v_o[__restrict__ ntotx], MULTIP(Int vq_o[__restrict__ ntotx],)
	ORDER2(Int v2_i[__restrict__ ntotx], Scalar v2_o[__restrict__ ntotx], MULTIP(Int vq2_o[__restrict__ ntotx],))
	// Position of current point
	const Int x_t[__restrict__ ndim], const Int x_i[__restrict__ ndim]
	){
	/* simplfication of arguments does not help
	const Scalar * u_t, 
	// Structure of the finite differences (input, output)
	const OffsetT offsets[nactx][ndim], 
	Int v_i[ntotx], Scalar v_o[ntotx], 
	// Position of current point
	const Int x_t[ndim], const Int x_i[ndim]
	){*/
#endif

	FACTOR(
	Scalar x_rel[ndim]; // Relative position wrt the seed.
	const bool factors = factor_rel(x_t,x_rel);
	)

	// Get the neighbor values, or their indices if interior to the block
	Int koff=0,kv=0; 
	for(Int kmix=0; kmix<nmix; ++kmix){
	for(Int kact=0; kact<nact; ++kact){
		const OffsetT * e = offsets[koff]; // e[ndim]
		++koff;
		SHIFT(
			Scalar fact[2]={0.,0.}; ORDER2(Scalar fact2[2]={0.,0.};)
			FACTOR( if(factors){factor_sym(x_rel,e,fact ORDER2(,fact2));} )
			DRIFT( const Scalar s = scal_vv(drift[kmix],e); fact[0]+=s; fact[1]-=s; )
			)

		for(Int s=0; s<2; ++s){
			if(s==0 && kact>=nsym) continue;
			OffsetT offset[ndim];
			const Int eps=2*s-1; // direction of offset
			mul_kv(eps,e,offset);
/*
			if(x_t[0]==16 && x_t[1]==28 && x_t[2]==8){
				printf("kv %i, v_i %i, v_o %f\n", kv, v_i[1],v_o[1]);
			}
*/
			WALLS(
			const bool visible = Visible(offset, x_t,wallDist_t, x_i,wallDist_i);
			if(!visible){
				v_i[kv]=-1; ORDER2(v2_i[kv]=-1;)
				v_o[kv]=infinity(); ORDER2(v2_o[kv]=infinity();)
				MULTIP(vq_o[kv]=0;  ORDER2(vq2_o[kv]=0;) )
				{++kv; continue;}
			})

			Int y_t[ndim], y_i[ndim]; // Position of neighbor. 
			add_vv(offset,x_t,y_t);
			add_vv(offset,x_i,y_i);

			if(local_i_macro && Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)))  {
				v_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v_o[kv] = fact[s];)
			} else {
				v_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot)) {
					const Int ny_t = Grid::Index_tot(y_t);
					v_o[kv] = u_t[ny_t] SHIFT(+fact[s]);
					MULTIP(vq_o[kv] = uq_t[ny_t];)
				} else {
					v_o[kv] = infinity();
					MULTIP(vq_o[kv] = 0;)
				}
			}
/*
			if(x_t[0]==16 && x_t[1]==28 && x_t[2]==8 && koff==2){
				printf("x_i %i,%i,%i, e %i,%i,%i, v_i %i\n",x_i[0],x_i[1],x_i[2],e[0],e[1],e[2], v_i[1]);
			}
*/
/*
			if(x_t[0]==16 && x_t[1]==28 && x_t[2]==8){
				printf("kv %i, v_i %i\n", kv, v_i[1]);
			}
*/
			ORDER2(
			add_vV(offset,y_t);
			add_vV(offset,y_i);

			if(local_i_macro && Grid::InRange(y_i,shape_i) PERIODIC(&& Grid::InRange(y_t,shape_tot)) ) {
				v2_i[kv] = Grid::Index(y_i,shape_i);
				SHIFT(v2_o[kv] = fact2[s];)
			} else {
				v2_i[kv] = -1;
				if(Grid::InRange_per(y_t,shape_tot) ) {
					const Int ny_t = Grid::Index_tot(y_t);
					v2_o[kv] = u_t[ny_t] SHIFT(+fact2[s]);
					MULTIP(vq2_o[kv] = uq_t[ny_t];)
				} else {
					v2_o[kv] = infinity();
					MULTIP(vq2_o[kv] = 0;)
				}
			}
			) // ORDER2

/*			if(x_t[0]==16 && x_t[1]==28 && x_t[2]==8){
				printf("kv %i,  v_i %i\n", kv, v_i[1]);
			}*/

			++kv;
		} // for s 
	} // for kact
	} // for kmix

#if !finitedifferences_inlinecode_macro
}
#endif