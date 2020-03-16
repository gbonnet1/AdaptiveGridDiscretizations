#pragma once

#ifndef Scalar_macro
typedef float Scalar;
#endif

#ifndef Int_macro
typedef int Int;
#endif

/// Ceil of the division of positive numbers
//Int ceil_div(Int num, Int den){return (num+den-1)/den;}


typedef unsigned char BoolPack;
typedef unsigned char BoolAtom;

// ----------- Flags -------------

/// A positive value may cause debug messages to be printed
#ifndef debug_print_macro
const Int debug_print = 0;
#endif

/** strict_iter_i_macro = 1 causes the input and output values 
within a block to be stored separately and synced at the end of 
each iteration*/
#ifndef strict_iter_i_macro
#define strict_iter_i_macro 0
#endif

/// Address float roundoff errors
#ifndef multi_precision_macro
#define multi_precision_macro 0
#endif

/// Source factorization
#ifndef factor_macro
#define factor_macro 0
#endif

