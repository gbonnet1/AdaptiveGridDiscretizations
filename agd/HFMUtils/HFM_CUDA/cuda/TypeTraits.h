#pragma once

#ifndef Scalar_macro
typedef float Scalar;
#endif

#ifndef Int_macro
typedef int Int;
#endif

typedef unsigned char BoolPack;
typedef unsigned char BoolAtom;

/* A positive value may cause debug messages to be printed*/
#ifndef debug_print_macro
const Int debug_print = 0;
#endif

/* strict_iter_i_macro = 1 causes the input and output values 
within a block are stored separately and synced at the end of 
each iteration*/
#ifndef strict_iter_i_macro
#define strict_iter_i_macro 0
#endif


Scalar infinity(){return 1./0.;}
Scalar not_a_number(){return 0./0.;}

/// Ceil of the division of positive numbers
Int ceil_div(Int num, Int den){return (num+den-1)/den;}
