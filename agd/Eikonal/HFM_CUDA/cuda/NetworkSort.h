#pragma once
// Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
// Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

/* This file implements efficient sorting methods, especially for small array sizes n<=32.
We use a network sort for n<=16, and merge sorts for larger n (n<=16384).
Note that these sorting methods are intended to be used on a single thread, hence
we do not attempt to extract parallelism.*/

namespace NetworkSort {
template<Int n, typename T,
Int rec = (n>16)+(n>32)+(n>64)+(n>128)+(n>256)+(n>512)+(n>1024)+(n>2048)+(n>4096)+(n>8192),
Int odd=rec%2 > struct Sorter;
}

// Please use only this function. All the rest is implementation detail
template<Int n, typename T> void network_sort(const T values[n], Int order[n]){
	NetworkSort::Sorter<n,T>::Run(values,order);}




namespace NetworkSort {
// ---- All below is implementation detail ----

template<Int n, typename T> struct NetworkSorter;

//template<Int n, typename T> void network_sort(const T values[n], Int order[n]);

template<Int n0, Int n1, Int n=n0+n1, typename T> 
void merge(const T values[n], const Int source[n], Int dest[n]){
	const Int * beg0 = source; const Int *end0 = source+n0;
	const Int * beg1 = end0;   const Int *end1 = source+n;
	for(Int i=0; i<n; ++i){
		if(beg0==end0 || (beg1!=end1 && values[*beg0]>values[*beg1]) ){
			  *dest=*beg1; ++beg1;}
		else {*dest=*beg0; ++beg0;}
		++dest;
	}
}


template<Int n,typename T> struct Sorter<n,T,0,0> {
	static void Run(const T values[n], Int order[n]){
		for(Int i=0; i<n; ++i){order[i]=i;}
		_Run(values, order);}
	static void _Run(const T values[n], Int order[n]){
		NetworkSorter<n,T>::Run(values, order);}
};
template<Int n,typename T> struct Sorter<n,T,1,1> {
	static void Run(const T values[n], Int order[n]){
		Int tmp[n]; for(Int i=0; i<n; ++i){tmp[i]=i;}
		_Run(values, tmp, order);}
	static void _Run(const T values[n], Int source[n], Int dest[n]){
		const Int n0=n/2; const Int n1=n-n0;
		Sorter<n0,T,0>::_Run(values,source);
		Sorter<n1,T,0>::_Run(values,source+n0);
		merge<n0,n1>(values, source, dest);
	}
};
template<Int n,typename T,Int r> struct Sorter<n,T,r,0> {
	static void Run(const T values[n], Int order[n]){
		Int tmp[n]; for(Int i=0; i<n; ++i){order[i]=i;}
		_Run(values, order, tmp);}
	static void _Run(const T values[n], Int order[n], Int tmp[n]){
		const Int n0=n/2; const Int n1=n-n0;
		Sorter<n0,T,r-1,1>::_Run(values,order,tmp);
		Sorter<n1,T,r-1,1>::_Run(values,order+n0,tmp+n0);
		merge<n0,n1>(values, tmp, order);
	}
};

template<Int n,typename T,Int r> struct Sorter<n,T,r,1> {
	static void Run(const T values[n], Int order[n]){
		Int tmp[n]; for(Int i=0; i<n; ++i){tmp[i]=i;}
		_Run(values, tmp, order);}
	static void _Run(const T values[n], Int source[n], Int dest[n]){
		const Int n0=n/2; const Int n1=n-n0;
		Sorter<n0,T,r-1,0>::_Run(values,source,dest); // dist is used as tmp
		Sorter<n1,T,r-1,0>::_Run(values,source+n0,dest+n0);
		merge<n0,n1>(values, source, dest);
	}
};

/*
Sorter<n,4> similar to Sorter<n,2> except for recursive call to Sorter<n,3>
Sorter<n,5> similar to Sorter<n,3> except for recursive call to Sorter<n,4>
 */

/*

The following sorting networks are 'best' according to the following website
http://pages.ripco.net/~jgamble/nw.html
The following license information can be found on that website as well.

This software is copyright (c) 2018 by John M. Gamble <jgamble@cpan.org>.
--- The GNU General Public License, Version 1, February 1989 ---


--- 2
[[0,1]]
--- 3
[[1,2]]
[[0,2]]
[[0,1]]
--- 4
[[0,1],[2,3]]
[[0,2],[1,3]]
[[1,2]]
--- 5
[[0,1],[3,4]]
[[2,4]]
[[2,3],[1,4]]
[[0,3]]
[[0,2],[1,3]]
[[1,2]]
--- 6
[[1,2],[4,5]]
[[0,2],[3,5]]
[[0,1],[3,4],[2,5]]
[[0,3],[1,4]]
[[2,4],[1,3]]
[[2,3]]
--- 7
[[1,2],[3,4],[5,6]]
[[0,2],[3,5],[4,6]]
[[0,1],[4,5],[2,6]]
[[0,4],[1,5]]
[[0,3],[2,5]]
[[1,3],[2,4]]
[[2,3]]
--- 8
[[0,1],[2,3],[4,5],[6,7]]
[[0,2],[1,3],[4,6],[5,7]]
[[1,2],[5,6],[0,4],[3,7]]
[[1,5],[2,6]]
[[1,4],[3,6]]
[[2,4],[3,5]]
[[3,4]]
--- 9
[[0,1],[3,4],[6,7]]
[[1,2],[4,5],[7,8]]
[[0,1],[3,4],[6,7],[2,5]]
[[0,3],[1,4],[5,8]]
[[3,6],[4,7],[2,5]]
[[0,3],[1,4],[5,7],[2,6]]
[[1,3],[4,6]]
[[2,4],[5,6]]
[[2,3]]
--- 10
[[4,9],[3,8],[2,7],[1,6],[0,5]]
[[1,4],[6,9],[0,3],[5,8]]
[[0,2],[3,6],[7,9]]
[[0,1],[2,4],[5,7],[8,9]]
[[1,2],[4,6],[7,8],[3,5]]
[[2,5],[6,8],[1,3],[4,7]]
[[2,3],[6,7]]
[[3,4],[5,6]]
[[4,5]]
--- 11
[[0,1],[2,3],[4,5],[6,7],[8,9]]
[[1,3],[5,7],[0,2],[4,6],[8,10]]
[[1,2],[5,6],[9,10],[0,4],[3,7]]
[[1,5],[6,10],[4,8]]
[[5,9],[2,6],[0,4],[3,8]]
[[1,5],[6,10],[2,3],[8,9]]
[[1,4],[7,10],[3,5],[6,8]]
[[2,4],[7,9],[5,6]]
[[3,4],[7,8]]
--- 12
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11]]
[[1,3],[5,7],[9,11],[0,2],[4,6],[8,10]]
[[1,2],[5,6],[9,10],[0,4],[7,11]]
[[1,5],[6,10],[3,7],[4,8]]
[[5,9],[2,6],[0,4],[7,11],[3,8]]
[[1,5],[6,10],[2,3],[8,9]]
[[1,4],[7,10],[3,5],[6,8]]
[[2,4],[7,9],[5,6]]
[[3,4],[7,8]]
--- 13
[[1,7],[9,11],[3,4],[5,8],[0,12],[2,6]]
[[0,1],[2,3],[4,6],[8,11],[7,12],[5,9]]
[[0,2],[3,7],[10,11],[1,4],[6,12]]
[[7,8],[11,12],[4,9],[6,10]]
[[3,4],[5,6],[8,9],[10,11],[1,7]]
[[2,6],[9,11],[1,3],[4,7],[8,10],[0,5]]
[[2,5],[6,8],[9,10]]
[[1,2],[3,5],[7,8],[4,6]]
[[2,3],[4,5],[6,7],[8,9]]
[[3,4],[5,6]]
--- 14
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
[[0,2],[4,6],[8,10],[1,3],[5,7],[9,11]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[3,7]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13]]
[[5,10],[6,9],[3,12],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]
--- 15
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13]]
[[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14]]
[[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]
--- 16
[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]]
[[0,2],[4,6],[8,10],[12,14],[1,3],[5,7],[9,11],[13,15]]
[[0,4],[8,12],[1,5],[9,13],[2,6],[10,14],[3,7],[11,15]]
[[0,8],[1,9],[2,10],[3,11],[4,12],[5,13],[6,14],[7,15]]
[[5,10],[6,9],[3,12],[13,14],[7,11],[1,2],[4,8]]
[[1,4],[7,13],[2,8],[11,14],[5,6],[9,10]]
[[2,4],[11,13],[3,8],[7,12]]
[[6,8],[10,12],[3,5],[7,9]]
[[3,4],[5,6],[7,8],[9,10],[11,12]]
[[6,7],[8,9]]

*/

#define ORDER_SWAP(i,j) { \
	const Int _i = order[i], _j=order[j]; \
	const bool b = values[_i]<values[_j]; \
	order[i] = b ? _i : _j; \
	order[j] = b ? _j : _i; \
} \

/* The rest of this file is generated by the following Python code applied to the 
above networks.

def code_line(s):
    s=s.replace("[[","ORDER_SWAP(")
    s=s.replace("],[",");ORDER_SWAP(")
    s=s.replace("]]",");")
    return s

def code_network(s):
    s = s.split("\n")
    n = int(s[0])
    s = (f"template<typename T> struct NetworkSorter<{n},T> " "{\n"
         f"static void Run(const T values[{n}],Int order[{n}])" "{\n"+
         "\n".join(code_line(l) for l in s[1:])+
         "}};\n")
    return s
def code_networks(s):
    s = s.split('--- ')[1:]
    return "\n".join(code_network(l) for l in s)

*/

template<typename T> struct NetworkSorter<2,T> {
static void Run(const T values[2],Int order[2]){
ORDER_SWAP(0,1);
}};

template<typename T> struct NetworkSorter<3,T> {
static void Run(const T values[3],Int order[3]){
ORDER_SWAP(1,2);
ORDER_SWAP(0,2);
ORDER_SWAP(0,1);
}};

template<typename T> struct NetworkSorter<4,T> {
static void Run(const T values[4],Int order[4]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);
ORDER_SWAP(1,2);
}};

template<typename T> struct NetworkSorter<5,T> {
static void Run(const T values[5],Int order[5]){
ORDER_SWAP(0,1);ORDER_SWAP(3,4);
ORDER_SWAP(2,4);
ORDER_SWAP(2,3);ORDER_SWAP(1,4);
ORDER_SWAP(0,3);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);
ORDER_SWAP(1,2);
}};

template<typename T> struct NetworkSorter<6,T> {
static void Run(const T values[6],Int order[6]){
ORDER_SWAP(1,2);ORDER_SWAP(4,5);
ORDER_SWAP(0,2);ORDER_SWAP(3,5);
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);
ORDER_SWAP(2,4);ORDER_SWAP(1,3);
ORDER_SWAP(2,3);
}};

template<typename T> struct NetworkSorter<7,T> {
static void Run(const T values[7],Int order[7]){
ORDER_SWAP(1,2);ORDER_SWAP(3,4);ORDER_SWAP(5,6);
ORDER_SWAP(0,2);ORDER_SWAP(3,5);ORDER_SWAP(4,6);
ORDER_SWAP(0,1);ORDER_SWAP(4,5);ORDER_SWAP(2,6);
ORDER_SWAP(0,4);ORDER_SWAP(1,5);
ORDER_SWAP(0,3);ORDER_SWAP(2,5);
ORDER_SWAP(1,3);ORDER_SWAP(2,4);
ORDER_SWAP(2,3);
}};

template<typename T> struct NetworkSorter<8,T> {
static void Run(const T values[8],Int order[8]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);
ORDER_SWAP(0,2);ORDER_SWAP(1,3);ORDER_SWAP(4,6);ORDER_SWAP(5,7);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(0,4);ORDER_SWAP(3,7);
ORDER_SWAP(1,5);ORDER_SWAP(2,6);
ORDER_SWAP(1,4);ORDER_SWAP(3,6);
ORDER_SWAP(2,4);ORDER_SWAP(3,5);
ORDER_SWAP(3,4);
}};

template<typename T> struct NetworkSorter<9,T> {
static void Run(const T values[9],Int order[9]){
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(6,7);
ORDER_SWAP(1,2);ORDER_SWAP(4,5);ORDER_SWAP(7,8);
ORDER_SWAP(0,1);ORDER_SWAP(3,4);ORDER_SWAP(6,7);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);ORDER_SWAP(5,8);
ORDER_SWAP(3,6);ORDER_SWAP(4,7);ORDER_SWAP(2,5);
ORDER_SWAP(0,3);ORDER_SWAP(1,4);ORDER_SWAP(5,7);ORDER_SWAP(2,6);
ORDER_SWAP(1,3);ORDER_SWAP(4,6);
ORDER_SWAP(2,4);ORDER_SWAP(5,6);
ORDER_SWAP(2,3);
}};

template<typename T> struct NetworkSorter<10,T> {
static void Run(const T values[10],Int order[10]){
ORDER_SWAP(4,9);ORDER_SWAP(3,8);ORDER_SWAP(2,7);ORDER_SWAP(1,6);ORDER_SWAP(0,5);
ORDER_SWAP(1,4);ORDER_SWAP(6,9);ORDER_SWAP(0,3);ORDER_SWAP(5,8);
ORDER_SWAP(0,2);ORDER_SWAP(3,6);ORDER_SWAP(7,9);
ORDER_SWAP(0,1);ORDER_SWAP(2,4);ORDER_SWAP(5,7);ORDER_SWAP(8,9);
ORDER_SWAP(1,2);ORDER_SWAP(4,6);ORDER_SWAP(7,8);ORDER_SWAP(3,5);
ORDER_SWAP(2,5);ORDER_SWAP(6,8);ORDER_SWAP(1,3);ORDER_SWAP(4,7);
ORDER_SWAP(2,3);ORDER_SWAP(6,7);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);
ORDER_SWAP(4,5);
}};

template<typename T> struct NetworkSorter<11,T> {
static void Run(const T values[11],Int order[11]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);
ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(9,10);ORDER_SWAP(0,4);ORDER_SWAP(3,7);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(4,8);
ORDER_SWAP(5,9);ORDER_SWAP(2,6);ORDER_SWAP(0,4);ORDER_SWAP(3,8);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(2,3);ORDER_SWAP(8,9);
ORDER_SWAP(1,4);ORDER_SWAP(7,10);ORDER_SWAP(3,5);ORDER_SWAP(6,8);
ORDER_SWAP(2,4);ORDER_SWAP(7,9);ORDER_SWAP(5,6);
ORDER_SWAP(3,4);ORDER_SWAP(7,8);
}};

template<typename T> struct NetworkSorter<12,T> {
static void Run(const T values[12],Int order[12]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);
ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);
ORDER_SWAP(1,2);ORDER_SWAP(5,6);ORDER_SWAP(9,10);ORDER_SWAP(0,4);ORDER_SWAP(7,11);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(3,7);ORDER_SWAP(4,8);
ORDER_SWAP(5,9);ORDER_SWAP(2,6);ORDER_SWAP(0,4);ORDER_SWAP(7,11);ORDER_SWAP(3,8);
ORDER_SWAP(1,5);ORDER_SWAP(6,10);ORDER_SWAP(2,3);ORDER_SWAP(8,9);
ORDER_SWAP(1,4);ORDER_SWAP(7,10);ORDER_SWAP(3,5);ORDER_SWAP(6,8);
ORDER_SWAP(2,4);ORDER_SWAP(7,9);ORDER_SWAP(5,6);
ORDER_SWAP(3,4);ORDER_SWAP(7,8);
}};

template<typename T> struct NetworkSorter<13,T> {
static void Run(const T values[13],Int order[13]){
ORDER_SWAP(1,7);ORDER_SWAP(9,11);ORDER_SWAP(3,4);ORDER_SWAP(5,8);ORDER_SWAP(0,12);ORDER_SWAP(2,6);
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,6);ORDER_SWAP(8,11);ORDER_SWAP(7,12);ORDER_SWAP(5,9);
ORDER_SWAP(0,2);ORDER_SWAP(3,7);ORDER_SWAP(10,11);ORDER_SWAP(1,4);ORDER_SWAP(6,12);
ORDER_SWAP(7,8);ORDER_SWAP(11,12);ORDER_SWAP(4,9);ORDER_SWAP(6,10);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(1,7);
ORDER_SWAP(2,6);ORDER_SWAP(9,11);ORDER_SWAP(1,3);ORDER_SWAP(4,7);ORDER_SWAP(8,10);ORDER_SWAP(0,5);
ORDER_SWAP(2,5);ORDER_SWAP(6,8);ORDER_SWAP(9,10);
ORDER_SWAP(1,2);ORDER_SWAP(3,5);ORDER_SWAP(7,8);ORDER_SWAP(4,6);
ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);
}};

template<typename T> struct NetworkSorter<14,T> {
static void Run(const T values[14],Int order[14]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(3,7);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
}};

template<typename T> struct NetworkSorter<15,T> {
static void Run(const T values[15],Int order[15]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(12,14);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(10,14);ORDER_SWAP(3,7);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);ORDER_SWAP(6,14);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(13,14);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(11,14);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
}};

template<typename T> struct NetworkSorter<16,T> {
static void Run(const T values[16],Int order[16]){
ORDER_SWAP(0,1);ORDER_SWAP(2,3);ORDER_SWAP(4,5);ORDER_SWAP(6,7);ORDER_SWAP(8,9);ORDER_SWAP(10,11);ORDER_SWAP(12,13);ORDER_SWAP(14,15);
ORDER_SWAP(0,2);ORDER_SWAP(4,6);ORDER_SWAP(8,10);ORDER_SWAP(12,14);ORDER_SWAP(1,3);ORDER_SWAP(5,7);ORDER_SWAP(9,11);ORDER_SWAP(13,15);
ORDER_SWAP(0,4);ORDER_SWAP(8,12);ORDER_SWAP(1,5);ORDER_SWAP(9,13);ORDER_SWAP(2,6);ORDER_SWAP(10,14);ORDER_SWAP(3,7);ORDER_SWAP(11,15);
ORDER_SWAP(0,8);ORDER_SWAP(1,9);ORDER_SWAP(2,10);ORDER_SWAP(3,11);ORDER_SWAP(4,12);ORDER_SWAP(5,13);ORDER_SWAP(6,14);ORDER_SWAP(7,15);
ORDER_SWAP(5,10);ORDER_SWAP(6,9);ORDER_SWAP(3,12);ORDER_SWAP(13,14);ORDER_SWAP(7,11);ORDER_SWAP(1,2);ORDER_SWAP(4,8);
ORDER_SWAP(1,4);ORDER_SWAP(7,13);ORDER_SWAP(2,8);ORDER_SWAP(11,14);ORDER_SWAP(5,6);ORDER_SWAP(9,10);
ORDER_SWAP(2,4);ORDER_SWAP(11,13);ORDER_SWAP(3,8);ORDER_SWAP(7,12);
ORDER_SWAP(6,8);ORDER_SWAP(10,12);ORDER_SWAP(3,5);ORDER_SWAP(7,9);
ORDER_SWAP(3,4);ORDER_SWAP(5,6);ORDER_SWAP(7,8);ORDER_SWAP(9,10);ORDER_SWAP(11,12);
ORDER_SWAP(6,7);ORDER_SWAP(8,9);
}};

} // Namespace NetworkSort
