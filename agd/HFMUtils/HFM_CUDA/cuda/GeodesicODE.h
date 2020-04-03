/**
This file implements a basic ODE solver, devoted to backtracking the minimal geodesics
using the upwind geodesic flow computed from an Eikonal solver.

(Note : since ODE integration is inherently a sequential process, it is admitedly a bit 
silly to solve it on the GPU. We do it here because the Python code is unacceptably slow,
and to avoid relying on compiled CPU code.)
*/

typedef int Int;
typedef float Scalar;
const Int ndim = 2;
const Int ncorners = 1<<ndim;
const bool periodic[ndim]={false,false};
__constant__ shape_tot[ndim];
__constant__ size_tot[ndim];

typedef unsigned char uchar;
const uchar uchar_MAX = 255;

__constant__ Int max_len = 200;
__constant__ Scalar causalityTolerance = 4; 
const Int min_len = 20;

/** Array suffix conventions:
- t : global field [physical dims][shape_tot]
_ s : data shared by all ODE solver threads [nThreads][len][physical dims]
*/

/** Computes the floor of the scalar components. Returns wether value changed.*/
bool Floor(const Scalar x[ndim], Int xq[ndim]){
	bool changed = false;
	for(Int k=0; k<ndim; ++k){
		const Int xqi = round(x[i]);
		if(xqi!=xq[i]) changed=true;
		xq[i]=xqi;
	}
	return changed;
}

/** This function estimates the flow at position x, by a bilinear interpolation of 
the flow at neighboring corners. Some corners are excluded from the interpolation, if the
associated distance value is judged to large. The neighbor flow values are reloaded 
only if necessary. Also returns the euclidean distance (or other) from the best corner to 
the target.*/
void Flow(const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const uchar * eucl_t, 
	const Scalar x[ndim], Int xq[ndim], 
	Scalar flow_cache[ncorners][ndim], bool exclude_cache[ncorners],
	Scalar flow[ndim], uchar * eucl){

	if(Floor(x,xq)){
		Scalar dist[ncorners];
		Scalar dist_min = infinity(); // Minimal distance among corners
		Scalar flow_weightsum; // flow weightsum for corner whose distance is minimal.
		for(Int icorner=0; icorner< ncorners; ++icorner){

			// Get the i-th corner and its index in the total shape.
			Int yq[ndim]; 
			for(Int k=0; k<ndim; ++k){yq[k] = xq[k]+((icorner >> k) & 1);}
			if(!InRange_per(yq,shape_tot)){
				exclude_cache[icorner]=true; 
				dist[icorner]=infinity(); 
				continue;}
			const Int ny = Index_per(yq,shape_tot);

			// Update the minimal distance, and corresponding weightsum, and eucl distance.
			dist[icorner] = dist_t[ny];
			if(dist[icorner]<dist_min){
				dist_min=dist[icorner];
				flow_weightsum = flow_weightsum_t[ny];
				*eucl = eucl_t[ny];
			}

			// Get the flow components
			for(Int k=0; k<ndim; ++k){
				flow_cache[icorner][k] = flow_vector_t[ny+size_tot*k];}
		}

		// Exclude interpolation neighbors with too high value.
		const Scalar dist_threshold = dist_min+causalityTolerance/flow_weightsum;
		for(Int icorner=0; icorner<ncorners; ++icorner){
			if(dist[icorner]>=dist_threshold){
				exclude_cache[icorner]=true;}
	}

	// Perform the interpolation
	Scalar wsum;
	for(Int k=0; k<ndim; ++k){flow[k]=0.;}

	for(Int icorner=0; icorner<ncorners; ++icorner){
		// Get the corner bilinear interpolation weight.
		if(exclude_cache[icorner]) continue;
		Scalar w = 1.;
		for(Int k=0; k<ndim; ++k){
			const Scalar dxk = x[k] - xq[k]
			w *= ((icorner>>k) & 1) ? 1-dxk : dxk;
		}

		// Add corner contribution
		wsum+=w;
		for(Int k=0; k<ndim; ++k){flow[k] += w*flow_cache[icorner][k];}
	}
	
	if(wsum>0){for(Int k=0; k<ndim; ++k){flow[k] /= wsum;}}
}


/*
void Flow(const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const uchar * eucl_t, 
	const Scalar x[ndim], Int xq[ndim], 
	Scalar flow_cache[ncorners][ndim], bool exclude_cache[ncorners],
	Scalar flow[ndim], uchar * eucl){
*/

extern "C" {

void GeodesicODE(const Scalar * flow_vector_t, const Scalar * flow_weightsum_t,
	const Scalar * dist_t, const uchar * eucl_t,
	Scalar * x_s, Int * len_s){

	const Int tid = BlockIdx.x * BlockDim.x + ThreadIdx.x;

	// Get the position, and euclidean distance to target, of the previous points
	Scalar x[min_len][ndim];
	const Int q_s_shape[3] = {BlockDim.x*GridDim.x, max_len, ndim};
	for(Int k=0; k<ndim; ++k){
		for(Int l=0; l<min_len; ++l){
			const Int q_s_pos[3]={tid,l,k};
			x[l][k] = x_s[Index(q_s_pos,q_s_shape)];
		}
	}
	uchar eucl[min_len];
	for(Int l=0; l<min_len; ++l){
		eucl[l] = eucl_t[tid*max_len + l];}

	Int xq[ndim]; for(Int k=0; k<ndim; ++k){xq[k]=Int_MAX;}
	Scalar flow_cache[ncorners][ndim]; 
	bool exclude_cache[ncorners];

	for(Int len = min_len-1; len<max_len; ++len){
		const Int l = len%min_len;
		const Int lNext = (len+1)%min_len;
		Scalar flow[ndim];
		Flow(flow_vector_t,flow_weightsum_t,dist_t,eucl_t,
			x[l],xq,flow_cache,exclude_cache,
			flow,eucl[lNext]);


	}





}

}