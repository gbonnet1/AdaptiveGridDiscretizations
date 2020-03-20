#pragma once

#include "TypeTraits.h"

const Int ndim = 2;

#ifndef shape_i_macro
const Int shape_i[ndim] = {24,24}; // Shape of a single block
const Int size_i = 24*24; // Product of shape_i
const Int log2_size_i = 10; // Upper bound on log_2(size_i)
#endif

#ifndef niter_i_macro
const Int niter_i = 48;
#endif

const Int offsets[ndim][ndim] = {{1,0},{0,1}};

#include "Isotropic_.h"
