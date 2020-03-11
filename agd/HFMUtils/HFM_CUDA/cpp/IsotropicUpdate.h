/*
In this notbook, we implement a GPU scheme for isotropic fast marching.
Further enhancements may include : 
- source factorization
- second order scheme
*/

/*
-- Convention --
The grid is split into blocks. We use the following naming convention : 
- x : absolute position
- x_i : position within a block (i=inner)
- x_o : position of a block (o=outer)
*/

/*
Notes on nvrtc
- "constexpr" not supported
- "using" not supported
- value "shape_i[0]*shape_i[1]" is not regarded as compile time constant
- !! Weird bug !! Using printf before a shared array allocation caused incomprehensible bugs.
*/

// Meta parameters. (We avoid templates, for faster compile, and easier debug.)
typedef float Scalar;
typedef int Int;
typedef char BoolPack; 

const Int ndim = 2;
const Int shape_i[ndim] = {8,8}; // Shape of a single block
const Int niter = 2;
const Int nsym = ndim; // Number of symmetric offsets

const Int size_i = 64; 
const Int log2_size_i = 6; // Upper bound on log_2(size_i)

const Int n_print = 100;
const Int n_print2=3;


Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}
//Scalar not_a_number()
//const Scalar infinity = __int_as_float(0x7f800000); //1./0.; //Scalar(1./0.);
//const Scalar not_a_number = 0.;///0.; //Scalar(0./0.);


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

Scalar _IsotropicUpdate(const Int n_i, const Scalar cost,
	const Scalar v_o[nsym][2], const Int v_i[nsym][2], const Scalar u_i[size_i]){

	// Get the minimal value among the right and left neighbors


	Scalar v[nsym];
	for(Int k=0; k<nsym; ++k){
		for(Int s=0; s<=1; ++s){
			const Int w_i = v_i[k][s];
			const Scalar v_ = w_i>=0 ? u_i[w_i] : v_o[k][s];
			v[k] = s==0 ? v_ : min(v_,v[k]);
			
			if(n_i==n_print2){
				printf("k%i,s%i, wi %i, v_ %f, v[k] %f\n",k,s,w_i,v_,v[k]);
				printf("u_i[2] %f, u_i[3] %f,  u_i[4] %f\n", u_i[2],u_i[3],u_i[4]);

			}
		}
	}


	// Sort the values (using a bubble sort)
	for(Int k=nsym-1; k>=1; --k){
		for(Int r=0; r<k; ++r){
			order(v,r);
		}
	}

	if(n_i==n_print2){
		printf("\n");
		for(Int k=0;k<nsym;++k){
			printf("v[%i]=%f\n",k,v[k]);
		}
		printf("value %f\n",u_i[n_i]);
	}

	// Compute the update
	const Scalar vmin = v[0];
	if(vmin==infinity()){return vmin;}
	for(Int k=1; k<nsym; ++k) {v[k]-=vmin;}
	Scalar value = cost;
	Scalar a=1., b=0., c = -cost*cost;
	for(Int k=1; k<nsym; ++k){
		const Scalar t = v[k];
		if(value<=t){
			if(n_i==n_print2) printf("value sent %f\n\n",vmin+value); 
			return vmin+value;}
		a+=1.;
		b+=t;
		c+=t*t;
		const Scalar delta = b*b-a*c;
		const Scalar sdelta = sqrt(delta);
		value = (b+sdelta)/a;
	}

	if(n_i==n_print2){
		printf("value solved %f\n\n",vmin+value);
	}

	return vmin+value;
}

/// Ceil of the division of positive numbers
Int ceil_div(Int num, Int den){return (num+den-1)/den;}

extern "C" {

__global__ void IsotropicUpdate(Scalar * u, const Scalar * metric, const BoolPack * seeds, const Int * shape,
	const Int * _x_o, Scalar * min_chg, const Scalar tol){

	// Setup coordinate system
	Int x_i[ndim], x_o[ndim], x[ndim]; 
	x_i[0] = threadIdx.x; x_i[1]=threadIdx.y; if(ndim==3) x_i[2]=threadIdx.z;
	const Int * __x_o = _x_o + ndim*blockIdx.x;
	for(int k=0; k<ndim; ++k){
		x_o[k] = __x_o[k];
		x[k] = x_o[k]*shape_i[k]+x_i[k];
	}

	GridType grid; // Share ?
	for(int k=0; k<ndim; ++k){
		grid.shape[k] = shape[k];
		grid.shape_o[k] = ceil_div(shape[k],shape_i[k]);
	}

	// Import local data
	const Int 
	n_i = grid.Index_i(x_i),
	n = grid.Index(x);

	const bool inRange = grid.InRange(x);
	const Scalar u_old = inRange ? u[n] : not_a_number();
	__shared__ Scalar u_i[size_i];
	__shared__ Scalar u_new[size_i];
	u_i[n_i] = u_old;
	u_new[n_i] = u_old;


	if(n==n_print2){
		printf("n_print = %i\n",n_print);
		printf("Grid shape %i %i, %i %i\n", 
			grid.shape[0],grid.shape[1],
			grid.shape_o[0],grid.shape_o[1]);

		printf("x_i %i %i \n", x_i[0], x_i[1]);
		printf("x_o %i %i \n", x_o[0], x_o[1]);
		printf("x %i %i \n", x[0], x[1]);
		printf("Hello world");

	}

	const Scalar cost = inRange ? metric[n] : not_a_number();
	const bool active = (cost < infinity()) && (! GetBool(seeds,n));

	if(n==n_print2){
		printf("inRange %i\n",inRange);
		printf("u_old %f\n",u_old);
		printf("cost %f\n",cost);
		printf("active %i\n",active);
		printf("u_i[n_i] %f\n",u_i[n_i]);

		printf("Seeds : %i %i %i %f %f\n",GetBool(seeds,0),GetBool(seeds,19),
			cost<infinity(),infinity(),not_a_number());
	}

	// Get the neighbor values, or their indices if interior to the block
	Scalar v_o[ndim][2];
	Int    v_i[ndim][2];
	for(Int k=0; k<ndim; ++k){
		for(Int s=0; s<2; ++s){
			Int * y = x; // Caution : aliasing
			Int * y_i = x_i;
			const Int eps=2*s-1;

			y[k]+=eps; y_i[k]+=eps;
			if(grid.InRange_i(y_i))  {v_i[k][s] = grid.Index_i(y_i);}
			else {
				v_i[k][s] = -1;
				if(grid.InRange(y)) {v_o[k][s] = u[grid.Index(y)];}
				else {v_o[k][s] = infinity();}
			}
			y[k]-=eps; y_i[k]-=eps;
		}
	}

	if(n==n_print2){
		for(Int k=0; k<ndim; ++k){
			for(Int s=0; s<ndim; ++s){
				printf("(k%i,s%i) v_o %f, v_i %i \n", k,s, v_o[k][s],v_i[k][s]);
			}
		}

	}

	// Make the updates
	for(int i=0; i<niter; ++i){
		if(active) {u_new[n_i] = _IsotropicUpdate(n_i, cost,v_o,v_i,u_i);}
		__syncthreads();
		u_i[n_i]=u_new[n_i];
		__syncthreads();
	}
	if(inRange){u[n] = u_i[n_i];}
	
	// Find the smallest value which was changed.
	Scalar u_diff = abs(u_old - u_i[n_i]);
	if( !(u_diff>tol) ){// Ignores NaNs (contrary to u_diff<=tol)
		u_i[n_i]=infinity();}

	if(n==0){
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
	}

	Int shift=1;
	for(Int k=0; k<log2_size_i; ++k){
		Int old_shift=shift;
		shift=shift<<1;
		if( (n_i%shift)==0){
			Int m_i = n_i+old_shift;
			if(m_i<size_i){
				u_i[n_i] = min(u_i[n_i],u_i[m_i]);
			}
		}
		if(k<log2_size_i-1) {__syncthreads();}
	}

	if(n==0){
		printf("u_i[0] %f,u_i[1] %f,u_i[2] %f\n",u_i[0],u_i[1],u_i[2]);
		printf("min_chg[0] %f\n",min_chg[0]);
	}
	if(n_i==0){min_chg[blockIdx.x] = u_i[0];
		printf("Hello world %f %i\n", u_i[0],blockIdx.x);
		printf("min_chg[0] %f\n",min_chg[0]);
	}

}

} // Extern "C"