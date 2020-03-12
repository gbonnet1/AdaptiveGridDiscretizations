#ifndef Scalar_macro
typedef float Scalar;
#endif

#ifndef Int_macro
typedef int Int;
#endif

typedef char BoolPack;

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
