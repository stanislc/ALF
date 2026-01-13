#! /usr/bin/env python

import sys, os
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Warning: numba not available, using regular numpy")

Nmc=5000000
Nl=1024
c=5.5
cutlsum=0.8
dl=1.0/Nl

Ndim=int(sys.argv[1])

if HAS_NUMBA:
    @jit(nopython=True, parallel=True)
    def compute_histogram_numba(Nmc, Ndim, Nl, c, cutlsum, seed):
        """Optimized histogram computation using numba"""
        np.random.seed(seed)
        histogram_a = np.zeros((Nl, 1))
        
        for i in prange(Nmc):
            ti = np.random.random(Ndim)
            unli = np.exp(c * np.sin(np.pi * ti - np.pi / 2))
            unli_sum = np.sum(unli)
            
            # Match original - no zero check
            li = unli / unli_sum
            
            for j in range(Ndim):
                for k in range(j + 1, Ndim):
                    lisum = li[j] + li[k]
                    if lisum > cutlsum:  # Match original condition exactly
                        ind = int(np.floor((li[j] / lisum) * Nl))
                        # Match original - no bounds checking, only basic safety
                        if 0 <= ind < Nl:
                            histogram_a[ind, 0] += 1
        
        return histogram_a

def compute_histogram_chunk(args):
    """Compute histogram for a chunk of Monte Carlo samples"""
    chunk_size, seed_offset, Ndim, Nl, c, cutlsum = args
    
    # Skip empty chunks
    if chunk_size <= 0:
        return np.zeros((Nl, 1))
    
    if HAS_NUMBA:
        return compute_histogram_numba(chunk_size, Ndim, Nl, c, cutlsum, seed_offset)
    else:
        # Fallback to regular numpy
        np.random.seed(seed_offset)
        local_histogram_a = np.zeros((Nl, 1))
        
        for i in range(chunk_size):
            ti = np.random.random_sample((Ndim,))
            unli = np.exp(c * np.sin(np.pi * ti - np.pi / 2))
            unli_sum = np.sum(unli)
            
            # Match original - no zero check
            li = unli / unli_sum
            
            for j in range(Ndim):
                for k in range(j + 1, Ndim):
                    lisum = li[j] + li[k]
                    if lisum > cutlsum:  # Match original condition exactly
                        ind = np.floor((li[j] / lisum) * Nl).astype('int')
                        # Match original - no bounds clipping, only basic safety
                        if 0 <= ind < Nl:
                            local_histogram_a[ind, 0] += 1
        
        return local_histogram_a

# Use all available CPU cores
num_cores = cpu_count()
print(f"Using {num_cores} CPU cores for parallel computation")
if HAS_NUMBA:
    print("Using numba JIT compilation for optimization")

# Ensure we have positive chunk sizes
if Nmc <= 0:
    print("Error: Nmc must be positive")
    sys.exit(1)
    
if num_cores <= 0:
    num_cores = 1

# Split work into chunks
chunk_size = max(1, Nmc // num_cores)  # Ensure minimum chunk size of 1
chunks = [(chunk_size, i * 1000, Ndim, Nl, c, cutlsum) for i in range(num_cores)]

# Handle remainder
remainder = Nmc % num_cores
if remainder > 0:
    chunks.append((remainder, num_cores * 1000, Ndim, Nl, c, cutlsum))

# Parallel computation
with Pool(num_cores) as pool:
    results = pool.map(compute_histogram_chunk, chunks)

# Combine results
histogram_a = np.zeros((Nl, 1))
for result in results:
    if result is not None:
        histogram_a += result

histogram = histogram_a + histogram_a[::-1]

# Add minimal regularization only if absolutely necessary to prevent crashes
histogram_safe = histogram + 1e-300  # Extremely small, won't affect results
S_k = np.log(histogram_safe)
G_k = -S_k + np.log(np.mean(histogram_safe))


# Save results
try:
    np.savetxt('G12_' + str(Ndim) + '.dat', G_k)
    print(f"Computation completed. Results saved to G12_{Ndim}.dat")
except Exception as e:
    print(f"Error saving results: {e}")
    sys.exit(1)
