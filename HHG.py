import numpy as np
import scipy
from tqdm import tqdm

from scipy.special import binom, xlogy

def get_tc(r_l, r_h, shapes, A, score):
    '''    
    input:
    r_l, r_h:   rank range of cell
    
    output: 
    t_c:        score of cell with rank range [r_l, r_h].
    '''

    N = shapes.sum()
    K = shapes.shape[0]
    w = r_h - r_l + 1 # width of cell 

    t_c = 0 # score of cell
    for k in range(K):
        e_c = w * shapes[k] / N
        o_c = A[k, r_h] - A[k, r_l-1]
        t_c += get_score(e_c, o_c, score)

    return t_c

def get_score(e_c, o_c, score):
    '''
    input:
    o_c, e_c:   observed, expected values
    score:      log-likelihood ('ll') or Pearson chi-square ('chi2')
    output:
    score (log-likelihood or Pearson chi-square) of the cell with given o_c, e_c
    '''
    if score=='ll':
        return xlogy(o_c, o_c/e_c)
    if score=='chi2':
        return (o_c-e_c)**2 / e_c

def C(n, k):
    '''
    binomial coefficient.
    '''
    if n < 0:
        return 0
    return binom(n, k)


def HHG_S(sample, m=3, score='ll'):
    '''
    input: 
    sample: list of numpy arrays with shape (N_i, ), N_1 + ... + N_K = N
    m:      int, number of cells in a partition
    score:  string, 'll' = log-likelihood or 'chi2' = Pearson chi-square score
    output:
    S_m:    HHG (agg by summation) statistic value.
    '''
    shapes = np.array([s.shape[0] for s in sample])
    stacked_sample = np.hstack(sample)
    stacked_ranks = scipy.stats.rankdata(stacked_sample).astype('int64') # no ties.

    ranks = np.split(stacked_ranks, shapes.cumsum()[:-1])

    N = shapes.sum() # total sample size
    K = len(sample)  # num of subsamples

    # for t_c calculations we need the special matrix A.
    # A[k, r] - numebr of observations with the rank (r=1,..,N) less than r from subsample k, A[k, 0] := 0
    A = np.zeros((K, N+1))

    for k in range(K):
        for r in range(N+1):
            A[k, r] = (ranks[k] <= r).sum()
    
    T_i = np.zeros(N+1) # T_i[w] = internal cells scores with width w, T_i[0] := 0
    T_e = np.zeros(N+1) # T_e[w] = edge cells scores with width w,     T_e[0] := 0

    # set of possible nodes edges: 0.5, 1.5,...,N+0.5
    nodes = np.arange(N+1) + 0.5

    for l in nodes:
        for r in nodes:
            if l >= r: 
                continue
            r_l = np.around(l+0.5).astype('int64')  # min rank in cell
            r_h = np.around(r-0.5).astype('int64')  # max rank in cell
            w = r_h - r_l + 1                       # width of cell                 

            if r_l==1 or r_h==N:
                T_e[w] += get_tc(r_l, r_h, shapes, A, score)
            else:
                T_i[w] += get_tc(r_l, r_h, shapes, A, score)
    
    S_m = 0
    for w in range(1, N+1):
        S_m += C(N-2-w, m-3)*T_i[w] + C(N-1-w, m-2)*T_e[w]

    return S_m
  
 
def HHG_M(sample, m=3, score='ll'):
    '''
    input: 
    sample: list of numpy array with shape (N_i, ), N_1 + ... + N_K = N
    m:      int, number of cells in partition
    score:  string, 'll' = log-likelihood or 'chi2' = Pearson chi-squared score
    output:
    M_m:    HHG (agg by maximization) statistic value.
    '''
    shapes = np.array([s.shape[0] for s in sample])
    stacked_sample = np.hstack(sample)
    stacked_ranks = scipy.stats.rankdata(stacked_sample).astype('int64') # no ties allowed.

    ranks = np.split(stacked_ranks, shapes.cumsum()[:-1])

    N = shapes.sum() # total sample size
    K = len(sample)  # num of samples

    A = np.zeros((K, N+1))

    for k in range(K):
        for r in range(N+1):
            A[k, r] = (ranks[k] <= r).sum()
    
    M = np.zeros((N+1, m+1))
    for i in range(1, N+1):
        w = i       # width of cell 
        r_l = 1     # min rank in cell
        r_h = i     # max rank in cell
        M[i, 1] = get_tc(r_l, r_h, shapes, A, score)

    for j in range(2, m+1):
        for i in range(j, N+1):
            max_val = -np.inf
            for a in range(j-1, i):
                val = M[a, j-1] + get_tc(a+1, i, shapes, A, score)
                if val > max_val:
                    max_val = val
            M[i, j] = max_val

    return M[N, m]


def ksample_permutation_test(sample, stat, N=100, **kwargs):
    '''
    input:
    sample: list of numpy array with shape (N_i, ), N_1 + ... + N_k = N
    stat:   test statistic (function) with additional parameters **kwargs
    N:      number of permutations
    '''
    shapes = np.array([s.shape[0] for s in sample])
    splits = shapes.cumsum()[:-1]
    stacked_sample = np.hstack(sample)
    vals = []
    val = stat(sample, **kwargs)

    for _ in range(N):
        np.random.shuffle(stacked_sample)
        shuffled = np.split(stacked_sample, splits)
        vals.append(stat(shuffled, **kwargs))

    vals = np.array(vals)
    pval = len(np.where(vals > val)[0]) / N
    return pval
