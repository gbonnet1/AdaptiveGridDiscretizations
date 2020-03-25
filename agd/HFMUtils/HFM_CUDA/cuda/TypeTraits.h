#pragma once

#ifndef Scalar_macro
typedef float Scalar;
#endif

#ifndef Int_macro
typedef int Int;
const Int IntMax = 2147483647;
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
#define strict_iter_i_macro 1
#endif

/** In multi-precision, we address float roundoff errors 
by representing a real in the form u+uq*multip_step, where
u is a float, uq is an integer, and multip_step is a constant.*/
#ifndef multiprecision_macro
#define multiprecision_macro 0
#endif

#if multiprecision_macro
#define MULTIP(...) __VA_ARGS__
#define NOMULTIP(...) 
#else
#define MULTIP(...) 
#define NOMULTIP(...) __VA_ARGS__
#endif

/** Source factorization allows to improve the solution accuracy by subtracting, before 
the finite differences computation, a expansion of the solution near the source.*/
#ifndef factor_macro
#define factor_macro 0
#endif

#if factor_macro
#define FACTOR(...) __VA_ARGS__
#define NOFACTOR(...) 
#else
#define FACTOR(...) 
#define NOFACTOR(...) __VA_ARGS__
#endif

/** A drift can be introduced in some schemes */
#ifndef drift_macro
#define drift_macro 0
#endif

#if drift_macro
#define DRIFT(...) __VA_ARGS__
#else
#define DRIFT(...) 
#endif

/** factorization and drift act similarly, by introducing a shift in the finite differences*/
#define shift_macro (factor_macro+drift_macro)

#if shift_macro
#define SHIFT(...) __VA_ARGS__
#else
#define SHIFT(...) 
#endif

/** The second order scheme allows to improve accuracy*/
#ifndef order2_macro
#define order2_macro 0
#endif

#if order2_macro
#define ORDER2(...) __VA_ARGS__
#else
#define ORDER2(...) 
#endif

/** Min or Max of a family of schemes*/
#ifndef mix_macro
#define mix_macro 0
#endif

#if mix_macro
#define MIX(...) __VA_ARGS__
#else
#define MIX(...) 
#endif

/** Curvature penalized models have share a few specific features : 
relaxation parameter, periodic boundary condition, xi and kappa constants, 
position dependent metric. */
#ifndef curvature_macro
#define curvature_macro 0
#endif

#if curvature_macro
#define CURVATURE(...) __VA_ARGS__
#define periodic_macro 1
#else
#define CURVATURE(...) 
#endif

/** Apply periodic boundary conditions on some of the axes.*/
#ifndef periodic_macro
#define periodic_macro 0
#endif

#if periodic_macro
#define PERIODIC(...) __VA_ARGS__
#define APERIODIC(...) 
#else
#define PERIODIC(...) 
#define APERIODIC(...) __VA_ARGS__
#endif
