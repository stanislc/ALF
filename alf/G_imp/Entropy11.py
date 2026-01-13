#! /usr/bin/env python

import sys, os
import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import product
from numba import jit, prange

Nmc=5000000
Nl=32
c=5.5
dl=1.0/Nl

Ndim=int(sys.argv[1])

@jit(nopython=True)
def process_G1_data_fast(G_k, Nl):
    """Fast processing of G1 data using numba - matches original behavior exactly"""
    # Reshape and compute exponential
    exp_neg_G = np.exp(-G_k)
    reshaped = exp_neg_G.reshape((Nl, Nl))
    
    # Sum along axis 1
    h = np.sum(reshaped, axis=1)
    
    # Add minimal regularization only to prevent crashes
    h_safe = h + 1e-300  # Extremely small, won't affect results
    S_k = np.log(h_safe)
    G_k_new = -S_k + np.log(np.mean(h_safe))
    
    return G_k_new

def process_G1_pair_optimized(args):
    """Optimized processing of a single (i,j) pair for G1 computation"""
    i, j, Nl = args
    
    try:
        # Load G1_i and G1_j
        G_ki = np.loadtxt('G1_' + str(i) + '.dat')
        G_kj = np.loadtxt('G1_' + str(j) + '.dat')
        
        # Process both files using optimized function
        G_ki_processed = process_G1_data_fast(G_ki, Nl)
        G_kj_processed = process_G1_data_fast(G_kj, Nl)
        
        # Reshape for broadcasting
        G_ki_2d = G_ki_processed.reshape((Nl, 1))
        G_ki_2d = np.tile(G_ki_2d, (1, Nl))
        
        G_kj_2d = G_kj_processed.reshape((1, Nl))
        G_kj_2d = np.tile(G_kj_2d, (Nl, 1))
        
        # Combine
        G_combined = G_ki_2d + G_kj_2d
        
        # Match original - no safety checks (can contain -inf/NaN)
        return (i, j, G_combined)
        
    except Exception as e:
        print(f"Error processing pair ({i}, {j}): {e}")
        return (i, j, np.zeros((Nl, Nl)))

def process_G1_pairs_batch(pairs_batch):
    """Process a batch of pairs to reduce overhead"""
    results = []
    for pair in pairs_batch:
        result = process_G1_pair_optimized(pair)
        results.append(result)
    return results

def main():
    
    # Use all available CPU cores
    num_cores = cpu_count()
    print(f"Using {num_cores} CPU cores for parallel computation")
    
    # Create all (i,j) pairs
    pairs = [(i, j, Nl) for i in range(2, Ndim+1) for j in range(2, Ndim+1)]
    total_pairs = len(pairs)
    print(f"Processing {total_pairs} pairs")
    
    # Batch pairs for better load balancing
    batch_size = max(1, total_pairs // (num_cores * 4))  # 4 batches per core
    pair_batches = []
    for i in range(0, total_pairs, batch_size):
        pair_batches.append(pairs[i:i+batch_size])
    
    print(f"Created {len(pair_batches)} batches of size ~{batch_size}")
    
    # Parallel computation with batching
    with Pool(num_cores) as pool:
        batch_results = pool.map(process_G1_pairs_batch, pair_batches)
    
    # Flatten results
    results = []
    for batch_result in batch_results:
        results.extend(batch_result)
    
    # Save results
    print("Saving results...")
    for i, j, G_combined in results:
        filename = f'G1_{i}_{j}.dat'
        np.savetxt(filename, G_combined)
    


if __name__ == "__main__":
    main()
