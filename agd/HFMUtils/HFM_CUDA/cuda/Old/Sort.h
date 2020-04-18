#pragma once

/*This file implements some very basic sorting algorithms.
We need to sort about 30 values in the worst case.
Yet the complexity of the bubble_sort is already prohibitive*/

// Standard bubble sort, complexity n*(n-1)/2
template<Int n>
void bubble_sort(const Scalar * values, Int * order){
		for(Int k=n-1; k>=1; --k){
		for(Int r=0; r<k; ++r){
			const Int i=order[r], j=order[r+1];
			if( values[i] > values[j] ){ 
				// swap( order[k], order[k+1] )
				const Int s = order[r];
				order[r] = order[r+1];
				order[r+1] = s;
			}
		}
	}
}


// Find the ordering of n values. n*log(n) complexity
template<int n>
void divide_sort(const Scalar * values, Int * order, Int * tmp){
	if(n<=6){return bubble_sort<n>(values,order);}
	const Int n0=n/2; const Int n1=n-n0;
	Int *order0=order, *order1=order+n0, * res = tmp;
	divide_sort<n0>(values,order0,tmp   );
	divide_sort<n1>(values,order1,tmp+n0); // Could use the same tmp, since non-parallel
	const Int * const end0 = order1, * const end1 = order+n;
	while(order0!=end0 && order1!=end1){
		if(values[*order0]<values[*order1]){*res=*order0; ++order0;} 
		else {*res=*order1; ++order1;}
		++res;
	}
	while(order0!=end0){*res=*order0; ++order0; ++res;}
	while(order1!=end1){*res=*order1; ++order1; ++res;}
	for(Int i=0; i<n; ++i){order[i]=tmp[i];}
}

/*
template<Int n>
void divide_sort_(const Scalar * values, Int * order, Int * tmp){
//	if(n<=6){return bubble_sort<n>(values,order);}
	const Int n0=n/2; const Int n1=n-n0;
	Int *order0=order, *order1=order+n0, *res = tmp;
	divide_sort_<n0>(values,order0,tmp);
	divide_sort_<n1>(values,order1,tmp+n0); // Could use the same tmp, since non-parallel
	Int i;

	const Int * const end0 = order1, * const end1 = order+n;
	for(i=0; i<n; ++i){
		if(values[*order0]<values[*order1]){
			  *res=*order0; ++order0; ++res; if(order0==end0) break;}
		else {*res=*order1; ++order1; ++res; if(order1==end1) break;}
	}
	++i;
	const Int r = n-i;
	Int *out = order+i;
	if(order0==end0){//std::cout << "first alt"<< std::endl;
		for(Int j=0; j<n1; ++j){if(j==r) break; *out=*order1; ++out; ++order1;}}
	else {//std::cout << "second alt, i="<< i << std::endl;
		for(Int j=0; j<n0; ++j){if(j==r) break; *out=*order0; ++out; ++order0; }}
	out=order;
	for(Int j=0; j<n && j<i; ++j){*out=*tmp; ++out; ++tmp; }
}

template<> void divide_sort_<1>(const Scalar * values, Int * order, Int *){
	return bubble_sort<1>(values, order);}
template<> void divide_sort_<2>(const Scalar * values, Int * order, Int *){
	return bubble_sort<2>(values, order);}
template<> void divide_sort_<3>(const Scalar * values, Int * order, Int *){
	return bubble_sort<3>(values, order);}
template<> void divide_sort_<4>(const Scalar * values, Int * order, Int *){
	return bubble_sort<4>(values, order);}
template<> void divide_sort_<5>(const Scalar * values, Int * order, Int *){
	return bubble_sort<5>(values, order);}
template<> void divide_sort_<6>(const Scalar * values, Int * order, Int *){
	return bubble_sort<6>(values, order);}
*/

/*
// Find the ordering of n values. n*log(n) complexity
template<Int n>
void divide_sort_(const Scalar * values, Int * order, Int * tmp){
	if(n<=6){return bubble_sort<n>(values,order);}
	const Int n0=n/2; const Int n1=n-n0;
	Int *order0=order, *order1=order+n0, *res = tmp;
	divide_sort_<n0>(values,order0,tmp);
	divide_sort_<n1>(values,order1,tmp+n0); // Could use the same tmp, since non-parallel
	Int i;
	const Int * const end0 = order1, * const end1 = order+n;
	for(i=0; i<n; ++i){
		if(values[*order0]<values[*order1]){*res=*order0; ++order0; if(order0==end0) break;} 
		else {*res=*order1; ++order1; if(order1==end1) break;}
		++res;
	}
	const Int r = n-i;
	Int *out = order+i; 
	if(order0==end0){for(Int j=0; j<n1 && j<r; ++j){*out=*order1; ++out; ++order1;}}
	else {for(Int j=0; j<n0 && j<r; ++j){*out=*order0; ++out; ++order0; }}
	out=order;
	for(Int j=0; j<n && j<i; ++j){*out=*order; ++out; ++order; }
}*/