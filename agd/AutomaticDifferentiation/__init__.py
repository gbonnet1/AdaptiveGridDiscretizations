# Copyright 2020 Jean-Marie Mirebeau, University Paris-Sud, CNRS, University Paris-Saclay
# Distributed WITHOUT ANY WARRANTY. Licensed under the Apache License, Version 2.0, see http://www.apache.org/licenses/LICENSE-2.0

from . import functional
from . import cupy_generic
from . import ad_generic
from . import misc
from . import Dense
from . import Sparse
from . import Reverse
from . import Dense2
from . import Sparse2
from . import Reverse2
from . import Optimization
from . import ad_specific

from .ad_generic import array,asarray,is_ad,remove_ad,common_cast,min_argmin, \
	max_argmax,disassociate,associate,apply_linear_mapping,apply_linear_inverse,precision

from .ad_specific import simplify_ad,apply,compose
from .cupy_generic import isndarray



