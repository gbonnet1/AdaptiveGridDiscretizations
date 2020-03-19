#pragma once

#include "TypeTraits.h"

const Int ndim = 2;
const Int nsym = 2; // Number of symmetric offsets
const Int nfwd = 0; // Number of forward offsets

const Int ntot = 4; // 2*nsym + nfwd
const Int nact = 2; // nsym + nfwd

#ifndef shape_i_macro
/*const Int shape_i[ndim] = {8,8}; // Shape of a single block
const Int size_i = 64; // Product of shape_i
const Int log2_size_i = 6; // Upper bound on log_2(size_i)*/
// Bigger seems better
const Int shape_i[ndim] = {24,24}; // Shape of a single block
const Int size_i = 576; // Product of shape_i
const Int log2_size_i = 10; // Upper bound on log_2(size_i)
#endif

#ifndef niter_i_macro
//const Int niter_i = 16;
const Int niter_i = 48;
#endif

const Int offsets[nact][ndim] = {{1,0},{0,1}};

#include "Isotropic_.h"
