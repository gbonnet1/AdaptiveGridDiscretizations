/** This file implements "dimension generic" (a.k.a dimension 4 and 5) tools 
for Voronoi's first reduction of quadratic forms.*/

namespace Voronoi {

struct SimplexStateT {
	Scalar m[symdim];
	Scalar a[ndim][ndim];
	Int vertex;
	Scalar objective;
//	m(m0),a(MatrixType::Identity()),vertex(-1),objective(infinity){}; 
};

void SetNeighbor(SimplexStateT & state,const Int neigh){
	// Record the new change of coordinates
//	const small * neigh_chg_flat = state.vertex==0 ? neigh_chg0[neigh] : neigh_chg1[neigh];
//	typedef const small (*smallMatrixT)[ndim];
//	const small (* neigh_chg)[ndim] = (smallMatrixT) neigh_chg_flat;
	Scalar a[ndim][ndim];  copy_aA(neigh_chg_[state.vertex][neigh],a); //copy_aA(neigh_chg_[state.vertex][neigh],a);
	Scalar sa[ndim][ndim]; copy_aA(state.a,sa); 
	dot_aa(a,sa,state.a);
	
	// Apply it to the reduced positive definite matrix
	Scalar sm[symdim]; copy_mM(state.m,sm); 
	Scalar ta[ndim][ndim]; trans_a(a,ta);
	gram_am(ta,sm,state.m);

	state.vertex = neigh_vertex_[state.vertex][neigh];
}

void KKT(const SimplexStateT & state, Scalar weights[kktdim], OffsetT offsets[kktdim][ndim]){
	const coefT coef       = coef_[state.vertex]; // coef[symdim][symdim]
	const supportT support = support_[state.vertex]; // support[kktdim][ndim]

	dim10::dot_av(coef,state.m,weights);
	Scalar aInv_[ndim][ndim]; inv_a(state.a,aInv_);
	Int aInv[ndim][ndim]; round_a(aInv_,aInv); // The inverse is known to have integer entries
	for(int i=0; i<kktdim; ++i){dot_av(aInv,support[i],offsets[i]);}

	KKT_Correct(state.vertex,weights);
}


void FirstGuess(SimplexStateT & state){
	state.objective = 1./0.; 
	for(int ivertex=0; ivertex<nvertex; ++ivertex){
		const Scalar obj = scal_mm(state.m,vertex_[ivertex]);
		if(obj>=state.objective) continue;
		state.vertex=ivertex;
		state.objective=obj;
	}
}

/** Returns a better neighbor, with a lower energy, for Voronoi's reduction.
If none exists, returns false*/
bool BetterNeighbor(SimplexStateT & state){
	const uchar * iw   = iw_[state.vertex];
	const uchar * stop = stop_[state.vertex];
	Scalar obj  = state.objective;
	Scalar bestObj=obj;
	int k=0, bestK = -1;
	const uchar * stopIt=stop; Int stop8=0;
	for(const uchar * iwIt=iw; iwIt!=iw; ++iwIt, ++stop8){
		if(stop8==8){stop8=0; ++stopIt;}
		uchar s = *iwIt;
		const int ind = int(s >> 4);
		s = s & 15;
		const Scalar wei = Scalar(s) - Scalar(s>=2 ? 1: 2);
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

const Int decompdim = Voronoi::kktdim;

void decomp_m(const Scalar m[symdim],Scalar weights[decompdim],OffsetT offsets[decompdim][ndim]){
	using namespace Voronoi;
	SimplexStateT state;
	copy_mM(m,state.m);
	identity_A(state.a);
	FirstGuess(state);
	for(Int i=0; i<maxiter; ++i){if(!BetterNeighbor(state)){break;}}
	KKT(state,weights,offsets);
}
