#pragma once

#include "TypeTraits.h"

const Int ndim = 3;

#ifndef shape_i_macro
const Int shape_i[ndim] = {4,4,4}; // Shape of a single block
const Int size_i = 4*4*4; // Product of shape_i
const Int log2_size_i = 6; // Upper bound on log_2(size_i)
#endif

#ifndef niter_i_macro
const Int niter_i = 8;
#endif

const Int offsets[ndim][ndim] = {{1,0,0},{0,1,0},{0,0,1}};

#include "Isotropic_.h"
