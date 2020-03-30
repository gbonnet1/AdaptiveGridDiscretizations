#pragma once

#include "static_assert.h"

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

/** Min or Max of a family of schemes*/
#ifndef mix_macro
#define mix_macro 0
#endif

#if mix_macro
#define MIX(...) __VA_ARGS__
#else
#define MIX(...) 
#endif

/** strict_iter_i_macro = 1 causes the input and output values 
within a block to be stored separately and synced at the end of 
each iteration*/
#ifndef strict_iter_i_macro
#define strict_iter_i_macro (multiprecision_macro || mix_macro)
#endif

/** strict_iter_o_macro causes a similar behavior, but for the global iterations */ 
#ifndef strict_iter_o_macro
#define strict_iter_o_macro multiprecision_macro
#endif

#if strict_iter_o_macro
#define STRICT_ITER_O(...) __VA_ARGS__
#else 
#define STRICT_ITER_O(...) 
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

/** Since the schemes are monotone (except with the second-order enhancement), and we start 
from a super-solution, the solution values should be decreasing as the iterations proceed. 
We can take advantage of this property to achieve better robustness of the solver. 
(Otherwise, floating point roundoff errors often cause multiple useless additional iterations)
*/
#ifndef decreasing_macro
#define decreasing_macro 1
#endif

#if decreasing_macro
#define DECREASING(...) __VA_ARGS__
#else 
#define DECREASING(...) 
#endif

/** The implemented schemes are causal except in the following cases:
- second order enhancement (mild non-causality)
- source factorization (mild non-causality)
- drift (strong non-causality, but possibly not such an issue due to the large block size)
We can take advantage of this property to improve computation time, by freezing computations
in the far future until the past has suitably converged. For that purpose, a target number of 
active blocks is specified. Blocks are then frozen, or not, depending on their minChg
(minimal change) value.
*/
#ifndef minChg_freeze_macro
#define minChg_freeze_macro 0
#endif

#if minChg_freeze_macro
#define MINCHG_FREEZE(...) __VA_ARGS__
#else
#define MINCHG_FREEZE(...) 
#endif

/** The pruning macro maintains a list of the active nodes at any time.
It is slightly more flexible than the default method, and needed for minChg_freeze_macro
to take effect.
*/
#ifndef pruning_macro
#define pruning_macro minChg_freeze_macro
#endif

#if pruning_macro
#define PRUNING(...) __VA_ARGS__
#else
#define PRUNING(...) 
#endif
