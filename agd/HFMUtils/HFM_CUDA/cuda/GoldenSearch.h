#pragma once

/*Scalar and Int types must be defined in enclosing file.*/

/** Usage : to minimize a function f on interval a,b.
const Scalar bounds[2] = {a,b};
Scalar midpoints[2]; golden_search::init(bounds,midpoints);
Scalar values[2]={f(midpoints[0]),f(midpoints[1])};
for(Int i=0; i<niter; ++i){
	const Int k = golden_search::step(midpoints,values);
	values[k] = f(midpoints[k]); 
}
*/

namespace golden_search {

const Scalar phi = (sqrt(5.)-1.)/2, psi = 1.-phi, delta = psi*psi/(phi-psi);

void init(const Scalar x[2], Scalar y[2]){
	y[0] = phi*x[0]+psi*x[1];
	y[1] = psi*x[1]+phi*x[1];
}
// returns the position where value needs to be updated
Int step(Scalar x[2], Scalar v[2], const bool mix_is_min=True){
	if(mix_is_min == (v[0]<v[1])){
		const Scalar x0 = x[0];
		x[0]-=(x[1]-x[0])*delta
		x[1]=x0;
		v[1]=v[0];
		return 0; 
	} else {
		const Scalar x1 = x[1];
		x[1]+=(x[1]-x[0])*delta;
		x[0]=x1;
		v[0]=v[1];
		return 1;
	}
}

} // golden_search