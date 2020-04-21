#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

#include "TypeTraits.h"
const Int ndim=4;
#include "Geometry_.h"

namespace Voronoi {

namespace dim10 {
	const Int ndim=10;
	#include "Geometry_.h"
}
/** This code is adapted from the c++ code in the CPU HFM library*/

// We implement below Voronoi's reduction of four dimensional positive definite matrices.
const Int maxiter=100;
const Int kktdim=12; // Number of support vectors in Voronoi's decomposition
typedef char small; // Small type to avoid overusing memory
typedef unsigned char uchar;
const Int nvertex = 2;
const small vertex_[nvertex][symdim] = { // The two four dimensional perfect forms
	{2, 1, 2, 1, 1, 2, 0, 1, 1, 2},
	{2, 1, 2, 1, 1, 2, 1, 1, 1, 2}
};

const Int nneigh[2] = {64,symdim}; // Number of neighbors of the perfect forms. 

const small neigh_vertex0[64] = { // The type of each neighbor as a perfect form
	1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,0,1,1,1,
	1,1,0,1,1,0,1,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1};
// The coordinate change from each neighbor to the corresponding perfect form
const small neigh_chg0[64][ndim*ndim] = { 
	{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1},{-1,0,0,0,0,-1,0,0,1,0,1,0,0,1,0,1},{-1,0,0,0,0,-1,0,0,1,0,1,0,0,1,0,1},{-1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1},{1,1,1,0,-1,0,0,0,0,0,-1,0,0,0,0,1},{1,0,1,0,-1,-1,0,0,0,0,-1,0,0,1,0,1},{1,0,1,0,0,-1,0,-1,0,0,-1,0,0,1,0,0},{-1,-1,-1,0,1,0,0,0,0,1,0,0,-1,0,0,1},{0,0,0,1,-1,-1,-1,0,1,0,0,-1,0,1,0,0},{1,1,0,0,0,-1,1,0,0,0,0,-1,0,0,-1,0},{-1,0,0,0,1,1,0,0,1,0,1,0,-1,0,0,1},{-1,0,-1,0,1,0,0,0,1,1,1,0,-1,0,0,1},{0,0,0,1,-1,-1,0,0,1,0,0,-1,1,1,1,0},{1,0,0,0,-1,-1,0,0,1,1,1,0,-1,0,0,1},{1,0,0,0,1,1,0,0,0,0,1,0,-1,0,0,1},{-1,0,0,0,0,-1,0,0,2,1,1,0,-1,0,0,1},{-1,0,0,0,1,1,0,0,1,0,1,0,0,0,0,1},{-2,-1,-1,0,1,0,0,0,1,1,0,0,-1,0,0,1},{1,0,0,0,0,1,0,0,1,0,1,0,-1,0,0,1},{-1,0,-1,0,0,0,-1,0,1,1,1,0,0,0,1,1},{0,0,-1,0,-1,-1,-1,0,1,0,1,0,0,1,1,1},{-1,-1,-1,0,1,0,0,0,0,1,0,0,0,0,1,1},{1,0,1,0,-1,-1,-1,-1,0,0,-1,0,0,1,0,0},{1,0,1,0,-1,-1,0,0,0,0,-1,0,0,1,1,1},{1,1,1,0,-1,0,0,0,0,0,-1,0,0,0,1,1},{1,1,1,1,0,-1,-1,-2,0,0,1,0,0,0,0,1},{1,1,1,1,0,0,1,1,0,-1,-1,-1,0,0,0,-1},{1,1,1,1,-1,0,0,1,0,-1,-1,-1,0,0,-1,-1},{1,1,0,0,0,0,1,1,-1,0,0,1,0,-1,-1,-1},{1,1,1,1,-1,0,0,1,0,-1,-1,-1,0,1,0,0},{1,1,1,1,-1,-1,0,0,0,-1,-1,-1,0,0,0,-1},{1,1,0,0,0,-1,1,0,-1,-1,-1,-1,0,0,-1,0},{1,1,1,1,-1,-1,-1,0,0,-1,-1,-1,0,1,0,0},
	{0,1,1,1,1,1,1,1,0,0,-1,-1,0,-1,0,-1},{1,1,1,1,-1,-1,0,0,0,-1,-1,-1,1,1,1,0},{-1,0,-1,0,1,0,0,0,1,1,1,0,0,0,1,1},{0,0,-1,0,-1,-1,-1,-1,1,0,1,0,1,1,1,0},{-2,-1,-1,0,1,0,0,0,1,1,0,0,-1,0,1,1},{1,1,0,0,-1,-1,-1,-1,0,-1,0,0,1,1,1,0},{1,0,0,0,-1,-1,0,0,1,1,1,0,0,1,0,1},{-1,0,0,0,0,-1,0,0,2,1,1,0,-1,1,0,1},{1,1,0,0,-1,-1,-1,0,0,-1,0,0,0,1,1,1},{-1,-1,0,0,0,-1,0,0,1,1,1,0,0,1,0,1},{1,1,0,0,-1,-1,-1,-1,0,-1,0,0,0,0,1,0},{-1,0,0,0,0,-1,0,0,1,1,1,0,0,1,0,1},{-1,0,0,0,0,-1,0,0,1,1,1,0,0,1,0,1},{1,1,0,0,-1,0,-1,0,0,-1,0,0,0,1,1,1},{1,0,0,-1,-1,-1,-1,-1,0,0,0,1,0,1,0,1},{-1,-1,-1,-1,1,0,0,0,0,1,0,0,0,-1,0,-1},{1,1,1,1,-1,0,0,0,0,0,-1,-1,0,-1,0,-1},{1,0,0,-1,-1,-1,-1,-1,0,0,0,1,0,1,0,0},{-1,-1,-1,-1,1,0,0,0,1,1,0,0,-1,-1,0,-1},{1,0,0,-1,-1,-1,0,0,0,0,0,1,0,1,1,1},{0,0,-1,-1,1,0,0,0,0,1,1,1,0,-1,0,-1},{1,0,0,-1,0,-1,0,-1,0,0,0,1,0,1,1,1},{0,0,0,-1,1,0,0,-1,0,1,0,1,0,0,1,1},{1,0,0,0,0,1,0,0,0,0,1,0,-1,0,0,1},{-1,0,0,0,0,-1,0,0,1,0,1,0,-1,1,0,1},{1,1,0,0,0,0,-1,-1,0,-1,0,0,0,0,1,0},{-1,0,0,0,0,-1,0,0,1,1,1,0,-1,0,0,1},{-1,0,0,0,1,1,0,0,0,0,1,0,0,0,0,1},{1,0,0,0,-1,-1,0,0,0,0,1,0,0,1,0,1},{1,1,0,0,-1,0,-1,0,0,-1,0,0,0,0,1,1},{-1,0,0,0,0,-1,0,0,1,1,1,0,0,0,0,1}
};

// All neighbors of the perfect form 1 are of type 0
const small neigh_vertex1[symdim] = {0,0,0,0,0,0,0,0,0,0}; 
const small neigh_chg1[symdim][ndim*ndim] = {
	{1,0,0,0,0,1,0,0,-1,0,1,0,1,0,0,1},{-1,0,0,0,0,-1,0,0,1,1,1,0,0,0,0,1},{-1,-1,-1,0,1,0,0,0,0,1,0,0,0,0,1,1},{-1,0,0,0,0,-1,0,0,1,1,1,0,0,1,0,1},{-1,-1,-1,-1,1,0,0,0,0,1,0,0,-1,-1,0,-1},{1,0,0,0,0,1,0,0,0,0,1,0,1,0,0,1},{-1,-1,-1,0,1,0,0,0,0,1,0,0,0,0,0,1},{1,0,0,0,-1,1,0,0,0,0,1,0,1,0,0,1},{-1,0,0,0,0,-1,0,0,1,0,1,0,0,1,0,1},{1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}
};

// This data enumerates the differences between the successive neighbors.
const uchar iw0[129] = {99,64,16,96,67,19,128,64,16,96,67,19,112,64,3,67,115,131,51,128,19,112,48,131,99,115,51,112,99,115,96,128,16,131,0,83,67,99,131,64,48,96,67,51,115,147,99,131,48,96,19,35,80,128,16,115,19,99,51,83,112,144,67,115,131,147,3,19,51,112,128,144,16,32,64,112,96,128,48,131,19,35,80,128,99,115,16,96,0,48,112,64,99,115,67,16,96,64,32,147,16,131,19,99,96,112,48,115,96,112,16,115,51,128,19,112,16,48,144,64,19,112,67,99,115,64,51,112,67};
const uchar stop0[17] = {219,54,170,170,218,166,162,98,130,168,168,108,171,170,202,182,1};		
const uchar iw1[37] = {48,51,112,51,67,83,115,131,19,35,48,80,115,128,16,32,64,99,131,147,3,19,51,112,128,144,0,16,48,96,128,16,131,19,64,67,96};
const uchar stop1 [5]= {133,32,8,66,21};

const small * neigh_vertex_[2] = {neigh_vertex0,neigh_vertex1};
//typedef const small * neigh_chgT[ndim*ndim];
//const neigh_chgT neigh_chg_[2] = {neigh_chg0,neigh_chg1};
const uchar * iw_[2]   = {iw0,iw1};
const uchar * stop_[2] = {stop0,stop1};

struct SimplexStateT {
	Scalar m[symdim];
	Scalar a[ndim][ndim];
	Int vertex;
	Scalar objective;
//	m(m0),a(MatrixType::Identity()),vertex(-1),objective(infinity){}; 
};


void SetNeighbor(SimplexStateT & state,const Int neigh){
	// Record the new change of coordinates
	const small * neigh_chg_flat = state.vertex==0 ? neigh_chg0[neigh] : neigh_chg1[neigh];
	//typedef (const small (*)[ndim]) smallMatrixT;
	typedef const small (*smallMatrixT)[ndim];
	const small (* neigh_chg)[ndim] = (smallMatrixT) neigh_chg_flat;
	Scalar a[ndim][ndim];  copy_aA(neigh_chg,a); //copy_aA(neigh_chg_[state.vertex][neigh],a);
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa); 
	dot_aa(a,sa,state.a);
	
	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm); 
	Scalar ta[ndim][ndim]; trans_a(a,ta);
	gram_am(ta,sm,state.m);

	state.vertex = neigh_vertex_[state.vertex][neigh];
}

const small coef0[symdim*symdim] = {1,1,0,1,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0};
const small support0[kktdim][ndim] = {
		{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},{1,0,0,-1},
		{1,0,-1,0},{1,-1,0,0},{0,1,0,-1},{0,1,-1,0},{0,0,1,-1},
		{0,0,0,0},{0,0,0,0}
	};
const small coef1[symdim*symdim] = {1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0
		,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,-1,0,0,-1,0,0,0,0,
		-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0
		,0,0,0,0,0,-1,0,-1,0,0,0,0,0,0,0,1,0,0,0};
const small support1[kktdim][ndim] = {
	{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1},{1,0,-1,0},{1,-1,0,0},
		{0,1,0,-1},{0,1,-1,0},{0,0,1,-1},{1,0,-1,1},{1,-1,0,1},{1,-1,-1,1}
	};
const small * coef_[2]={coef0,coef1};
const (small (*)[ndim]) support_[2] = {support0,support1};


void KKT(const SimplexStateT & state, Scalar weights[kktdim], OffsetT offsets[kktdim][ndim]){
	const small coef = coef_[state.vertex];
	const small support = support_[state.vertex];

	dim10::dot_av(coef,state.m,weights);
	Scalar aInv_[ndim][ndim]; inv_a(state.a,aInv_);
	Int aInv[ndim][ndim]; round_a(aInv_,aInv); // The inverse is known to have integer entries
	for(int i=0; i<kktdim; ++i){dot_av(aInv,support[i],offsets[i]);}

	// A bit of post processing is needed to get non-negative weights
	if(state.vertex==1){
		for(int i=symdim; i<kktdim; ++i){weights[i] = 0.;}
	} else {
		// Compute a non-negative solution
		const Scalar
		l0 = -min(min(min(0.,weights[1]),weights[4]),weights[8]),
		l1 = -min(min(min(0.,weights[0]),weights[3]),weights[7]),
		u01= min(min(min(weights[2],weights[5]),weights[6]),weights[9]);
		
		// Triangle of decompositions is defined by the inequalities c0>=l0, c1>=l1, c0+c1<=u01
		// Check that non-empty //assert(l0+l1<=u01);
		
		// Vertices are (l0,l1), (l0,u01-l0), (u01-l1,l1)
		// We use their use the barycenter
		// Indeed, this triangle is mapped to the space of decompositions by a linear map with small integer entries. As a result, the angles are bounded away from zero.
		// (The triangle cannot degenerate to a segment.)
		const Scalar c[2] = { (2*l0+u01-l1)/3., (2*l1+u01-l0)/3.};
		const ScalarType mc01 = -(c[0]+c[1]);
		weights[0]+=c[1];
		weights[1]+=c[0];
		weights[2]+=mc01;
		weights[3]+=c[1];
		weights[4]+=c[0];
		weights[5]+=mc01;
		weights[6]+=mc01;
		weights[7]+=c[1];
		weights[8]+=c[0];
		weights[9]+=mc01;
		weights[10]=c[0];
		weights[11]=c[1];
	}
}

// ----- Dimension generic functions --------

void decomp_m(const Scalar m[symdim],Scalar weights[kktdim],OffsetT offsets[kktdim][ndim]){
	SimplexStateT state;
	copy_mM(m,state.m);
	for(Int i=0; i<ndim; ++i){for(Int j=0; j<ndim; ++j){state.a[i][j]=(i==j);
	FirstGuess(state);
	for(Int i=0; i<maxiter; ++i){if(!BetterNeighbor(state)){break;}}
	KKT(state,weights,offsets);
}

void FirstGuess(SimplexStateType & state){
	state.objective = infinity(); 
	for(int ivertex=0; ivertex<nvertex; ++ivertex){
		const ScalarType obj = scal_mm(state.m,vertex_[ivertex]);
		if(obj>=state.objective) continue;
		state.vertex=vertex;
		state.objective=obj;
	}
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateType & state){
	const small * iw   = iw_[state.vertex];
	const small * stop = stop_[state.vertex];
	ScalarType obj  = state.objective;
	ScalarType bestObj=obj;
	int k=0, bestK = -1;
	small * stopIt=stop; Int stop8=0;
	for(small * iwIt=iw; iwIt!=iw; ++iwIt, ++stop8){
		if(stop8==8){stop8=0; ++stopIt;}
		small s = *iwIt;
		const int ind = int(s >> 4);
		s = s & 15;
		const ScalarType wei = Scalar(s) - Scalar(s>=2 ? 1: 2);
		obj += wei*state.m[ind];
		if(!(((*stopIt)>>stop8)&1)) continue;
		if(obj<bestObj) {
			bestObj=obj;
			bestK=k;}
		++k;
	}
	if(bestK==-1) return false;
	state.objective=bestObj; // Note : roundoff error could be an issue ?
	SetNeighbor(state,bestK); // neighs[bestK]
	return true;
}



} // namespace Voronoi