import numpy as np
from numba import cuda

# Define the CUDA kernel for merging
@cuda.jit
def merge_kernel(ids, pair, idx, new_ids, length):
    i = cuda.grid(1)
    if i < length:
        if i < length - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids[i] = idx
            if i + 1 < length:
                new_ids[i + 1] = -1  # Mark the next index as processed
        elif ids[i] != -1:
            new_ids[i] = ids[i]

def merge_gpu(ids, pair, idx):
    # Convert inputs to NumPy arrays
    ids = np.array(ids, dtype=np.int32)
    new_ids = np.zeros_like(ids) - 1  # Initialize new_ids with -1 to mark empty spots
    pair = np.array(pair, dtype=np.int32)

    # Transfer data to GPU
    d_ids = cuda.to_device(ids)
    d_new_ids = cuda.to_device(new_ids)
    d_pair = cuda.to_device(pair)
    d_idx = idx

    # Define grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (len(ids) + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    merge_kernel[blocks_per_grid, threads_per_block](d_ids, d_pair, d_idx, d_new_ids, len(ids))

    # Ensure kernel execution is completed
    cuda.synchronize()

    # Copy result back to host
    new_ids = d_new_ids.copy_to_host()

    # Remove the markers (-1) and return the result
    result_ids = []
    skip = False
    for id_value in new_ids:
        if id_value == -1:
            continue
        if skip:
            skip = False
            continue
        result_ids.append(id_value)
        if id_value == d_idx:
            skip = True

    return result_ids

# ids = [9, 3, 14, 15, 11, 7, 17, 1, 3, 6, 17, 1, 18, 12, 17, 4, 5, 11, 13, 18, 7, 10, 2, 7, 14, 14, 17, 10, 3, 19]
# pair = (17, 1)
# idx = 21
# merged_ids = merge_gpu(ids, pair, idx)
# print(merged_ids)


@cuda.jit
def get_stats_kernel(tokens, counts, vocab_size):
    i = cuda.grid(1)
    if i < len(tokens) - 1:
        # Encode pair as a unique integer index
        pair_index = tokens[i] * vocab_size + tokens[i + 1]
        cuda.atomic.add(counts, pair_index, 1)

def get_stats_gpu(tokens):
    # Convert tokens to a NumPy array
    tokens = np.array(tokens, dtype=np.int32)

    # Determine vocabulary size
    vocab_size = max(tokens) + 1

    # Initialize counts array (size = vocab_size * vocab_size)
    counts_size = vocab_size * vocab_size
    counts = np.zeros(counts_size, dtype=np.int32)

    # Transfer data to GPU
    d_tokens = cuda.to_device(tokens)
    d_counts = cuda.to_device(counts)

    # Define grid and block dimensions
    threads_per_block = 256
    blocks_per_grid = (tokens.size + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    get_stats_kernel[blocks_per_grid, threads_per_block](d_tokens, d_counts, vocab_size)

    # Ensure kernel execution is completed
    cuda.synchronize()

    # Copy result back to host
    counts = d_counts.copy_to_host()
    
    # Convert the flat array back to a dictionary
    counts_dict = {}
    for i in range(counts_size):
        if counts[i] > 0:
            pair = (i // vocab_size, i % vocab_size)
            counts_dict[pair] = counts[i]
    
    return counts_dict

# Example usage
# tokens = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2]
# stats = get_stats_gpu(tokens)
# print("Stats: ", stats)
# print("Max stats", max(stats, key=stats.get))
