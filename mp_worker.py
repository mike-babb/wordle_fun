# mike babb
# mp_worker.py
# worker functions for the multiprocessing version of the
# "five groups of five, 25 different letters" solver.
#
# These must live in a real module (not defined inline in the notebook)
# so that they can be pickled and sent to worker processes. This matters
# most on macOS/Windows, where multiprocessing uses the 'spawn' start
# method by default -- functions defined in a notebook's __main__ can't
# be pickled there. On Linux (the default 'fork' method) it would work
# either way, but keeping the workers in a module makes the code portable.

import numpy as np

# ---------------------------------------------------------------------
# Per-process globals. These are populated once per worker by
# _init_worker() and then reused for every chunk that worker handles,
# instead of being re-sent (and re-pickled) with every single task.
# ---------------------------------------------------------------------
_l2_all = None
_word_byte_array = None
_w1b_arr = None
_w2b_arr = None
_l2_arr = None


def _init_worker(l2_all_arr, word_byte_array_arr, w1b_arr, w2b_arr, l2_arr):
    """
    Pool initializer: runs once in each worker process and stashes the
    large, read-only arrays in module-level globals so every task on
    that worker can reuse them without re-transmitting them.
    """
    global _l2_all, _word_byte_array, _w1b_arr, _w2b_arr, _l2_arr
    _l2_all = l2_all_arr
    _word_byte_array = word_byte_array_arr
    _w1b_arr = w1b_arr
    _w2b_arr = w2b_arr
    _l2_arr = l2_arr


def process_chunk(index_range):
    """
    Do the same work the original single-threaded loop did, but only
    for l2_df rows [start_idx, end_idx).

    Returns an (n, 5) int32 array of [w1b, w2b, l2, w3bw4b, w5b] rows,
    matching the columns of the original 'total_output' array.
    """
    start_idx, end_idx = index_range

    l2_all = _l2_all
    word_byte_array = _word_byte_array
    w1b_arr = _w1b_arr
    w2b_arr = _w2b_arr
    l2_arr = _l2_arr

    chunks = []

    for i_row in range(start_idx, end_idx):
        w1b = w1b_arr[i_row]
        w2b = w2b_arr[i_row]
        l2 = l2_arr[i_row]

        # bitwise and to identify l2 values with no letters in common with l2
        positional_idx_l2l3l4 = (l2_all & l2) == 0

        if not positional_idx_l2l3l4.any():
            continue

        # l2 values with different letters than l2 above (this is l4)
        output_array_w3bw4b = l2_all[positional_idx_l2l3l4]

        # l2, l3, l4 accumulated letters
        output_array_l2l3l4 = output_array_w3bw4b | l2

        for w3bw4b, l2l3l4 in zip(output_array_w3bw4b, output_array_l2l3l4):
            # check against the word_byte_array for w5b
            positional_idx_l2l3l4l5 = (word_byte_array & l2l3l4) == 0

            if not positional_idx_l2l3l4l5.any():
                continue

            # the final five letters not in the group of twenty
            output_array_l2l3l4l5 = word_byte_array[positional_idx_l2l3l4l5]
            n_rows_l2l3l4l5 = output_array_l2l3l4l5.shape[0]

            temp_output = np.empty(shape=(n_rows_l2l3l4l5, 5), dtype=np.int32)
            temp_output[:, 0] = w1b
            temp_output[:, 1] = w2b
            temp_output[:, 2] = l2
            temp_output[:, 3] = w3bw4b
            temp_output[:, 4] = output_array_l2l3l4l5

            chunks.append(temp_output)

    if chunks:
        return np.vstack(chunks)
    else:
        return np.empty(shape=(0, 5), dtype=np.int32)


# ---------------------------------------------------------------------
# Level-2 build: all no-letters-in-common pairs of words.
# Same idea as above -- chunk on the outer index of the combinations()
# iteration and let each worker walk its own slice.
# ---------------------------------------------------------------------
_word_byte_list = None


def _init_l2_worker(word_byte_list_arr):
    global _word_byte_list
    _word_byte_list = word_byte_list_arr


def process_l2_chunk(index_range):
    """
    Reproduce combinations(word_byte_list, 2) for outer indices
    [start_idx, end_idx), i.e. all pairs (i, j) with start_idx <= i <
    end_idx and i < j < n. Returns an (n, 3) int32 array of
    [w1b, w2b, l2] rows, matching the original l2_list columns.
    """
    start_idx, end_idx = index_range
    word_byte_list = _word_byte_list
    n = len(word_byte_list)

    rows = []
    for i in range(start_idx, end_idx):
        w1_be = word_byte_list[i]
        for j in range(i + 1, n):
            w2_be = word_byte_list[j]
            if w1_be & w2_be == 0:
                rows.append((w1_be, w2_be, w1_be | w2_be))

    if rows:
        return np.array(rows, dtype=np.int32)
    else:
        return np.empty(shape=(0, 3), dtype=np.int32)
