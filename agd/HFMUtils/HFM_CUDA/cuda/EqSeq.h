/*
This file contains a family of 'well equidistributed' sequences in the 
integers {0,...,n-1}. (Aka low discrepancy sequences.)
(Generated using a personal approach, since I could not find a suitable reference
for this particular problem.)
*/

template<int n> int eqseq[n];
template<> int eqseq<2>[2] = {0,1};
template<> int eqseq<3>[3] = {1,0,2};
template<> int eqseq<4>[4] = {1,2,0,3};
template<> int eqseq<5>[5] = {2,1,4,0,3};
template<> int eqseq<6>[6] = {2,4,0,3,5,1};
template<> int eqseq<7>[7] = {3,1,5,2,6,0,4};
template<> int eqseq<8>[8] = {3,5,1,6,2,4,0,7};
template<> int eqseq<9>[9] = {4,2,7,1,6,3,8,0,5};
template<> int eqseq<10>[10] = {4,7,1,6,2,9,3,5,0,8};
template<> int eqseq<11>[11] = {5,2,9,4,7,0,8,3,10,1,6};
template<> int eqseq<12>[12] = {5,8,1,10,3,6,2,9,4,11,0,7};
template<> int eqseq<13>[13] = {6,3,10,1,9,5,12,2,7,4,11,0,8};
template<> int eqseq<14>[14] = {6,10,2,8,4,12,0,9,5,13,3,7,1,11};
template<> int eqseq<15>[15] = {7,3,12,5,10,1,13,6,9,0,11,4,14,2,8};
template<> int eqseq<16>[16] = {7,11,2,13,4,9,1,14,6,10,3,12,5,8,0,15};
template<> int eqseq<17>[17] = {8,4,14,2,11,7,15,1,10,5,13,3,12,6,16,0,9};
template<> int eqseq<18>[18] = {8,13,2,11,5,16,3,10,6,15,0,12,7,17,1,9,14,4};
template<> int eqseq<19>[19] = {9,4,15,7,13,1,17,5,11,2,16,8,12,3,14,6,18,0,10};

/*
# Python code to generate the contents of this file.
import numpy as np
def wasserstein(a,b):
    """Return the L2 wasserstein distance between 1D measures equidistributed at the given points"""
    a,b = np.sort(np.repeat(a,len(b))), np.sort(np.repeat(b,len(a)))
    return ((a-b)**2).sum()/len(a)

def nextpoint(a,n):
    """Choose the next point to insert in a which makes it closest to range(n) in wasserstein distance"""
    return np.argmin(list(wasserstein(a+[i],range(n)) if i not in a else np.inf for i in range(n)))

def eqseq(n):
    """Greedy ordering to approximate range(n) as well as possible as quickly as possible"""
    a=[]
    for i in range(n): 
        a = a+[nextpoint(a,n)]
    return a

def display(l):
    n=len(l)
    print(f"template<> eqseq<{n}>[{n}] = {{{','.join(map(str,l))}}};")
*/