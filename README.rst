HHG (Heller-Heller-Gorfine k-sample tests).
===========================================

Written on python with the use of ``numpy`` and ``scipy``. For the details see the article *"Consistent Distribution-Free K-Sample and Independence Tests for
Univariate Random Variables"* by Ruth Heller, Yair Heller, Shachar Kaufman, Barak Brill and Malka Gorfine.

Example.
-------

.. code:: python

    from HHG import HHG_S, HHG_M, ksample_permutation_test
    import numpy as np
    import scipy.stats
    
    n = 50  # subsample size
    N = 250 # permutations
    m = 3   # splits for HHG
    
    X = scipy.stats.norm.rvs(size=n)
    Y = scipy.stats.norm.rvs(size=n, loc=0.5)
    
    # we can get permutation pvalues
    pval_S = ksample_permutation_test([X, Y], HHG_M, N=N, m=m)
    pval_M = ksample_permutation_test([X, Y], HHG_S, N=N, m=m)
    
    # and values of statistics
    val_S = HHG_S([X, Y], m=m, score='ll')
    val_M = HHG_M([X, Y], m=m, score='ll')

