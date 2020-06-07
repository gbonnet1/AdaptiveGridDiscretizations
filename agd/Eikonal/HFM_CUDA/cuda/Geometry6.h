#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"

#define ndim_macro 6
const int ndim=ndim_macro;
#include "Geometry_.h"
#include "Inverse_.h"
#include "NetworkSort.h"

#define CUDA_DEVICE // Do not include <math.h>
#include "LinProg/Siedel_Hohmeyer_LinProg.h"

namespace Voronoi {

/** This code is adapted from the c++ code in the CPU HFM library*/
namespace dim_symdim {
	const Int ndim=symdim;
	#include "Geometry_.h"
}

typedef char small; // Small type to avoid overusing memory
typedef unsigned char uchar;

#include "Geometry6/Geometry6_data.h"
#include "Geometry6/Geometry6_datag.h"
#include "Geometry6/Geometry6_datakkt.h"

#ifndef GEOMETRY6_DATA2//This file is a bit huge, so it is not embedded in the agd library
#include "../../../../Miscellaneous/Geometry6_data2.h"
#endif


struct SimplexStateT {
	Scalar m[symdim];
	Scalar a[ndim][ndim];
	Int vertex;
	Scalar objective;
};


/** This code extends the 5-dimensional code to handle senary forms */

// We implement below Voronoi's reduction of four dimensional positive definite matrices.
const Int maxiter=100;

// The seven six dimensional perfect forms, vertices of Ryskov's polyhedron
const Int nvertex = 7;

const Scalar vertex_[nvertex][symdim] = {
 {2. ,1. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,0.5,0.5,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0.5,1. ,2. ,1. ,1. ,1. ,2. ,1. ,0.5,1. ,1. ,2. ,0.5,1. ,1. ,0.5,0. ,2. }
};

// ------ For GroupElem ------

// Number of neighbors of the perfect forms.
const Int nneigh_[nvertex] = {21, 6336, 38124, 21, 621, 46, 21};
// Number of classes of neighbors of each perfect form
const int nneigh_base_[7] = {1, 8, 11, 3, 3, 5, 1} ;
// The number of active constraints, at each perfect form
const int nsupport_[7] = {21, 30, 36, 21, 27, 22, 21} ;
typedef const small (*vertex_supportT)[6]; // small[][6]
const int ndiff_[nvertex] = {ndiff0,ndiff1,ndiff2,ndiff3,ndiff4,ndiff5,ndiff6};
typedef const small (*keyT)[symdim]; // small[][symdim]
typedef const small (*kkt_2weightsT)[symdim]; // small[symdim][symdim]


// ----- Group all those things togeter ----
struct vertex_dataT {
	const Scalar * vertex;
	
	// ------ For GroupElem ------
	
	// Number of neighbors
	const Int nneigh;
	// The class of each neighbor vertex
	const uchar * neigh_vertex;
	// The next two encode the change of variable from neighbor toward reference form
	const unsigned int * neigh_choice;
	const uchar * neigh_signs;
	
	// Number of classes of neighbors of each perfect form
	const int nneigh_base;
	// The vertex type of each neighbor class
	const int * neigh_base_v;
	// The change of variables from the neighbor, to the reference perfect form
	const chgi_jT * neigh_base_c;
	
	// The number and the list of the active constraints, at this vertex
	const int nsupport;
	const vertex_supportT vertex_support;
	
	// ----- For Better neighbor ------

	// The number of elementwise differences between the successive neighbors
	const int ndiff;
	// One key neighbor is placed every 1024, to avoid roundoff error accumulation
	const keyT key;
	// The place where successive neighbors differ
	const uchar * diff_i;
	// By how much the successive neighbors differ, at the given place
	const small * diff_v;
	
	// ----- For KKT -----

	const kkt_2weightsT kkt_2weights;
	const kkt_constraintsT kkt_constraints;
	
} vertex_data_[nvertex] = {
	{vertex_[0], nneigh_[0],neigh_vertex0,neigh_choice0,neigh_signs0, nneigh_base_[0],neigh0_base_v,neigh0_base_c, nsupport_[0],vertex_support0, ndiff0,key0,diff0_i,diff0_v, kkt_2weights0,kkt_constraints0},
	{vertex_[1], nneigh_[1],neigh_vertex1,neigh_choice1,neigh_signs1, nneigh_base_[1],neigh1_base_v,neigh1_base_c, nsupport_[1],vertex_support1, ndiff1,key1,diff1_i,diff1_v, kkt_2weights1,kkt_constraints1},
	{vertex_[2], nneigh_[2],neigh_vertex2,neigh_choice2,neigh_signs2, nneigh_base_[2],neigh2_base_v,neigh2_base_c, nsupport_[2],vertex_support2, ndiff2,key2,diff2_i,diff2_v, kkt_2weights2,kkt_constraints2},
	{vertex_[3], nneigh_[3],neigh_vertex3,neigh_choice3,neigh_signs3, nneigh_base_[3],neigh3_base_v,neigh3_base_c, nsupport_[3],vertex_support3, ndiff3,key3,diff3_i,diff3_v, kkt_2weights3,kkt_constraints3},
	{vertex_[4], nneigh_[4],neigh_vertex4,neigh_choice4,neigh_signs4, nneigh_base_[4],neigh4_base_v,neigh4_base_c, nsupport_[4],vertex_support4, ndiff4,key4,diff4_i,diff4_v, kkt_2weights4,kkt_constraints4},
	{vertex_[5], nneigh_[5],neigh_vertex5,neigh_choice5,neigh_signs5, nneigh_base_[5],neigh5_base_v,neigh5_base_c, nsupport_[5],vertex_support5, ndiff5,key5,diff5_i,diff5_v, kkt_2weights5,kkt_constraints5},
	{vertex_[6], nneigh_[6],neigh_vertex6,neigh_choice6,neigh_signs6, nneigh_base_[6],neigh6_base_v,neigh6_base_c, nsupport_[6],vertex_support6, ndiff6,key6,diff6_i,diff6_v, kkt_2weights6,kkt_constraints6},
};


/** Generates an isometry for the given vertex, 
which puts the corresponding neighbor in reference position.
Returns the index of the reference form.
*/
int GroupElem(const int ivertex, const int neighbor,
	small g[__restrict__ ndim][ndim]){
	const vertex_dataT & data = vertex_data_[ivertex];
	const int nsupport = data.nsupport;
	const uchar edge = data.neigh_vertex[neighbor]; //unsigned to silence warning
	uint choice = data.neigh_choice[neighbor];
	char sign = data.neigh_signs[neighbor];

	/*
	std::cout << "choice " << choice << " and sign" << int(sign) << std::endl;
	std::cout << "ivertex " <<ivertex << std::endl;
*/
	// Decompose the choice and signs
	uint choices[ndim]; 	small signs[ndim];
	for(int i=0; i<ndim; ++i){
		choices[i] = choice % nsupport;
		choice /= nsupport;
		signs[i] = 1-2*(sign % 2);
		sign /= 2;
	}

	// Build the change of variables from the support vectors
	small g0[ndim][ndim];
	for(int j=0; j<ndim; ++j){
		const uint k = choices[j];
		const small * v = data.vertex_support[k];
//		show_v(std::cout,v);
		const small s = signs[j];
//		std::cout << k << " " << int(s) << std::endl;
		for(int i=0; i<ndim; ++i){
			g0[i][j] = s*v[i];}
	}
/*
	std::cout << "choices and signs" << std::endl;
	show_v(std::cout, choices);
	show_v(std::cout, signs);
	std::cout << "g0 " << std::endl;
	show_a(std::cout, g0); std::cout << std::endl;*/
	// If necessary, compose with the base change of variables
	chgi_jT chg = data.neigh_base_c[edge];
	if(chg==nullptr){copy_aA(g0,g);}
	else {dot_aa(chg,g0,g);}

	return data.neigh_base_v[edge];
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateT & state){
	const int ivertex = state.vertex;
	const vertex_dataT & data = vertex_data_[ivertex];

	Scalar obj = dim_symdim::scal_vv(state.m,data.key[0]);
	int best_neigh = 0;
	Scalar best_obj = obj;
	for(int idiff=0,ineigh=1; idiff<data.ndiff; ++idiff){
		const uchar index = data.diff_i[idiff];
		obj += data.diff_v[idiff] * state.m[index & 31];
		if(index & 32){ // Completed neighbor
			if(obj<best_obj){
				best_obj = obj;
				best_neigh = ineigh;
			}
			++ineigh;
		}
	}

	// Now set that neighbor in the state, if necessary
	if(best_obj>=state.objective) return false;
	state.objective = best_obj;

	// Record the new vertex
	small a_[ndim][ndim];
	state.vertex = GroupElem(ivertex,best_neigh,a_);
	
	// Record the new change of coordinates
	Scalar a[ndim][ndim]; copy_aA(a_,a); // cast to scalar to avoid small overflow
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa); 
	dot_aa(a,sa,state.a);

	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm); 
	tgram_am(a,sm,state.m);

	return true;
}

void KKT(const SimplexStateT & state, Scalar weights[symdim], 
	OffsetT offsets[symdim][ndim]){
	const vertex_dataT data = vertex_data_[state.vertex];
	
	// Compute a decomposition, possibly with negative entries
	dim_symdim::dot_av(data.kkt_2weights,state.m,weights);
	dim_symdim::div_Vk(weights, 2);
	
	// Change of variables toward original coordinates.
	Scalar aInv_[ndim][ndim]; inv_a(state.a,aInv_);
	Int aInv[ndim][ndim]; round_a(aInv_,aInv);

	// Number of minimal vectors for the perfect form
	const int nsupport = data.nsupport;
	const int nsupport_max = 36; // Upper bound
	OffsetT offsets_[nsupport_max][ndim]; // Using [nsupport][ndim]
	for(int i=0; i<nsupport; ++i){dot_av(aInv,data.vertex_support[i],offsets_[i]);}


	
	if(nsupport==symdim){
		// Case where the vertex is non-degenerate.
		// There is only one possible decomposition, and it must be non-negative.
		for(int i=0; i<symdim; ++i){copy_vV(offsets_[i],offsets[i]);}
		return;
	} else {
		// Case where the vertex is degenerate.
		// Solve a linear program to find a non-negative decomposition
		
		// Dimension of the linear program
		const int d = nsupport - symdim;
		const int d_max = nsupport_max - symdim;
				
		// ---- Define the half spaces intersections. (linear constraints) ----
		Scalar halves[(nsupport_max+1)*(d_max+1)]; // used as Scalar[nsupport+1][d+1];
		
		Scalar maxWeight = 0;
		for(Int i=0; i<symdim; ++i) maxWeight = max(maxWeight,abs(weights[i]));
		if(maxWeight==0) maxWeight=1;
		
		// The old components must remain positive
		for(int i=0; i<symdim; ++i){
			for(int j=0; j<d; ++j){
				halves[i*(d+1)+j] = data.kkt_constraints[j][i];}
			halves[i*(d+1)+d] = weights[i]/maxWeight;
		}
		
		// The new components must be positive
		for(int i=symdim; i<nsupport; ++i){
			for(int j=0; j<d; ++j){
				halves[i*(d+1)+j] = (i-symdim)==j;}
			halves[i*(d+1)+d] = 0;
		}
		
		// Projective component positive
		for(int j=0; j<d; ++j){halves[nsupport*(d+1)+j] = 0;}
		halves[nsupport*(d+1)+d] = 1;
						
		// Minimize some arbitrary linear form (we only need a feasible solution)
		Scalar n_vec[d_max+1]; // used as Scalar[d+1]
		Scalar d_vec[d_max+1]; // used as Scalar[d+1]
		for(int i=0; i<d; ++i) {n_vec[i]=1; d_vec[i]=0;}
		n_vec[d]=1; d_vec[d]=1;
			
		Scalar opt[d_max+1];
		const int size = nsupport+1;
		const int size_max = nsupport_max+1;
		Scalar work[((size_max+3)*(d_max+2)*(d_max-1))/2]; // Scalar[4760]
		
		const int BadIndex = 1234567890;
		int next[size_max] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
			21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37};
		int prev[size_max] = {BadIndex,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,
			18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35};
#ifdef DOUBLE
		dlinprog
#else
		slinprog
#endif
		(halves, 0, size, n_vec, d_vec, d, opt, work, next, prev, size_max);
		// TODO : check that status is correct

		// The solution is "projective". Let's normalize it, dividing by the last coord.
		for(int i=0; i<d; ++i){opt[i]/=opt[d];}

		// Get the solution, and find the non-zero weights, which should be positive.
		Scalar sol[nsupport_max]; // Using sol[nsupport]
		for(int i=0; i<symdim; ++i){
			Scalar s=0;
			for(int j=0; j<d; ++j) {s+=opt[j]*halves[i*(d+1)+j];}
			s*=maxWeight;
			sol[i]=s+weights[i];
		}
		for(int i=0; i<d; ++i){sol[symdim+i] = maxWeight*opt[i];}
		// We only need to exclude the d smallest elements. For simplicity, we sort all.
		Int ord[nsupport_max], tmp[nsupport_max]; // using Int[nsupport]
		variable_length_sort(sol, ord, tmp, nsupport);

		for(int i=0; i<symdim; ++i){
			const int j=ord[i+d];
			weights[i] = sol[j];
			copy_vV(offsets_[j],offsets[i]);
		} // for i
		
		
	}
}


} // Namespace Voronoi

const Int decompdim=symdim;
#include "Voronoi_.h"
