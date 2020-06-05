#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0


#include "TypeTraits.h"
const Int ndim=6;
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

struct SimplexStateT {
	Scalar m[symdim];
	Scalar a[ndim][ndim];
	Int vertex;
	Scalar objective;
};


/** This code extends the 5-dimensional code to handle senary forms */

// We implement below Voronoi's reduction of four dimensional positive definite matrices.
const Int maxiter=100;
const Int kktdim=36; // Number of support vectors in Voronoi's decomposition
const Int decompdim=symdim;
const Int nvertex = 7;
const Scalar vertex_[nvertex][symdim] = { // The seven six dimensional perfect forms
 {2. ,1. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,1. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0. ,1. ,2. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,1. ,1. ,2. ,1. ,1. ,1. ,1. ,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,0.5,0.5,0.5,2. },
 {2. ,0.5,2. ,1. ,1. ,2. ,1. ,1. ,0.5,2. ,1. ,1. ,0.5,0.5,2. ,1. ,1. ,1. ,1. ,1. ,2. },
 {2. ,0. ,2. ,0.5,1. ,2. ,1. ,1. ,1. ,2. ,1. ,0.5,1. ,1. ,2. ,0.5,1. ,1. ,0.5,0. ,2. }
};

// ------ For GroupElem ------

// Number of neighbors of the perfect forms. (Guess which is the wicked one.)
const Int nneigh[nvertex] = {21, 144, 38124, 21, 621, 46, 21}; 

// The class of the neighbor vertex of a perfect form, in the list.
const uchar * neigh_vertex[nvertex] = 
{neigh_vertex0,neigh_vertex1,neigh_vertex2,neigh_vertex3,neigh_vertex4,neigh_vertex5,neigh_vertex6};
// The change of variable toward that perfect form.
const int * neigh_choice[nvertex] = 
{neigh_choice0,neigh_choice1,neigh_choice2,neigh_choice3,neigh_choice4,neigh_choice5,neigh_choice6};
const uchar * neigh_signs[nvertex] = 
{neigh_signs0,neigh_signs1,neigh_signs2,neigh_signs3,neigh_signs4,neigh_signs5,neigh_signs6};


// Number of classes of neighbors of each perfect form
const int nneigh_base[7] = {1, 8, 11, 3, 3, 5, 1} ;
// The vertex type of each neighbor class
const int * neigh_base_v[7] = 
{neigh0_base_v,neigh1_base_v,neigh2_base_v,neigh3_base_v,neigh4_base_v,neigh5_base_v,neigh6_base_v};
// The change of variables from the neighbor, to the reference perfect form
const chgi_jT * neigh_base_c[7] = 
{neigh0_base_c,neigh1_base_c,neigh2_base_c,neigh3_base_c,neigh4_base_c,neigh5_base_c,neigh6_base_c};


// The number of active constraints, at each perfect form
const int nvertex_support[7] = {21, 30, 36, 21, 27, 22, 21} ;
typedef small (*vertex_supportT)[6]; // small[][6]
const vertex_supportT vertex_support = 
{vertex_support0,vertex_support1,vertex_support2,vertex_support3,vertex_support4,vertex_support5,vertex_support6};


// ----- For Better neighbor ------

// The number of elementwise differences between the successive neighbors
const int ndiff[nvertex] = {ndiff0,ndiff1,ndiff2,ndiff3,ndiff4,ndiff5,ndiff6}
// The number of key neighbors (a single one, except for form 2 to avoid roundoff errors)
const int nkey[nvertex] = {nkey0,nkey1,nkey2,nkey3,nkey4,nkey5,nkey6};

// Some key neighbors are given fully, to avoid roundoff error accumulation
typedef const small (*keyT)[symdim]; // small[][symdim]
const keyT key[nvertex] = {key0,key1,key2,key3,key4,key5,key6};

// The place where successive neighbors differ
const uchar * diff_i[nvertex] = {diff0_i,diff1_i,diff2_i,diff3_i,diff4_i,diff5_i,diff6_i};
// By how much the successive neighbors differ, at the given place
const small * diff_v[nvertex] = {diff0_v,diff1_v,diff2_v,diff3_v,diff4_v,diff5_v,diff6_v};

// ----- For KKT -----

typedef small (*kkt_2weightsT)[symdim]; // small[symdim][symdim]
const kkt_2weightsT kkt_2weights[nvertex] = 
{kkt_2weights0,kkt_2weights1,kkt_2weights2,kkt_2weights3,kkt_2weights4,kkt_2weights5,kkt_2weights6};
typedef small (*kkt_constraintsT)[symdim]; // small[][symdim]
const kkt_constraintsT kkt_constraints[nvertex] = 
{kkt_constraints0,kkt_constraints1,kkt_constraints2,kkt_constraints3,kkt_constraints4,kkt_constraints5,kkt_constraints6};
const int * kkt_offsets[nvertex] = 
{kkt_offsets0,kkt_offsets1,kkt_offsets2,kkt_offsets3,kkt_offsets4,kkt_offsets5,kkt_offsets6};

/** Generates an isometry for the given vertex, 
which puts the corresponding neighbor in reference position.
Returns the index of the reference form.
*/
int GroupElem(const int vertex, const int neighbor,
	small g[__restrict__ dim][dim]){
	const vertex_supportT support = vertex_support[vertex];
	const int nsupport = nvertex_support[vertex];

	const char edge = neigh_vertex[vertex][neighbor];
	const uint choice = neigh_choice[vertex][neighbor];
	const char sign = neigh_signs[vertex][neighbor];

	// Decompose the choice and signs
	uint choices[dim]; 	small signs[dim];
	for(int i=0; i<dim; ++i){
		choices[i] = choice % nsupport;
		choice /= nsupport;
		signs[i] = 1-(sign % 2);
		sign /= 2
	}

	// Build the change of variables from the support vectors
	small g0[dim][dim];
	for(int j=0; j<dim; ++j){
		const uint k = choices[j];
		const small * v = support[k];
		const small s = signs[k];
		for(int j=0; j<ndim; ++j){
			g0[i][j] = s*v[i];}
	}

	// If necessary, compose with the base change of variables
	chgi_jT chg = neigh_base_c[vertex][edge];
	if(chg==nullptr){copy_aA(g0,g);}
	else {dot_aa(chg,g0,g);}

	return neigh_base_v[vertex][edge];
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateT & state){
	const int vertex = state.vertex;
	const uchar * index = diff_i[vertex];
	const small * value = diff_v[vertex];
	const keyT * keys = key[vertex]
	const int ndifferences = ndiff[vertex];

	Scalar obj = dim_symdim::scal_vv(state.m,keys[0]);
	int best_neigh = 0;
	Scalar best_obj = obj;
	for(int idiff=0,ineigh=1; idiff<ndiff; ++idiff){
		obj += value[idiff] * state.m[index[idiff & 31]];
		if(idiff & 32){ // Completed neighbor
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

	// Record the new change of coordinates
	Scalar a[ndim][ndim];  
	state.vertex = GroupElem(vertex,best_neigh,a);
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa); 
	dot_aa(a,sa,state.a);

	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm); 
	tgram_am(a,sm,state.m);

	return true;
}

void KKT(const SimplexStateT & state, Scalar weights[symdim], 
	OffsetT offsets[symdim][ndim]){
	const int vertex = state.vertex;
	const vertex_supportT support = vertex_support[vertex];
	// Compute a decomposition, possibly with negative entries
	dim_symdim::dot_av(kkt_2weights[vertex],state.m,weights);
	const chgi_jT a = state.a; // inverse ?

	if(nvertex_support[vertex]==symdim){
		for(int i=0; i<symdim; ++i){dot_av(a,support[i],offsets[i]);
	} else {
		// Solve the linear system to find a non-negative decomposition

	}
}


} // Namespace Voronoi
