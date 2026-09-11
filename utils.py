# standard
import pickle
import os
from string import ascii_lowercase
from itertools import product, permutations, combinations

# external
import pandas as pd
import numpy as np
import networkx as nx

# load input data

def load_input_data(all_unique_letters:bool = True):

    # load the data
    word_df = pd.read_csv(filepath_or_buffer=  'words_alpha.txt', header = None, names = ['word'], dtype = str)
    # format as str
    word_df['word'] = word_df['word'].astype(str)
    # convert to lowercase
    word_df['lcase'] = word_df['word'].str.lower()
    # count letters
    word_df['n_letters'] = word_df['word'].str.len()
    # sort letters
    word_df['letters_sorted'] = word_df['lcase'].map(lambda x: ''.join(sorted(x)))
    # create a set
    word_df['lcase_set'] = word_df['lcase'].map(lambda x: set(x))
    # count unique letters
    word_df['n_unique_chars'] = word_df['lcase_set'].map(lambda x: len(x))


    # drop duplicates
    word_df = word_df.drop_duplicates(subset = ['letters_sorted']).copy()
    
    

    # byte encode words
    word_df['word_byte'] = word_df['word'].map(byte_encode_words)

    if all_unique_letters:
        word_df = word_df.loc[(word_df['n_unique_chars'] == 5) & (word_df['n_letters'] == 5), :]    

    # sort and create a new index
    word_df = word_df.sort_values(by = 'lcase').reset_index(drop = True)
    # word id
    word_df['word_id'] = range(0, word_df.shape[0])

    # word id list
    word_id_list = word_df['word_id'].to_numpy(dtype = np.int16)

    # word byte list, array, dict
    word_byte_list = word_df['word_byte'].tolist()
    word_byte_array = np.array(word_byte_list, dtype = np.int32)
    word_byte_to_word_dict = {wb:lcase for wb, lcase in zip(word_df['word_byte'], word_df['lcase'])}

        
    return word_df, word_id_list, word_byte_list, word_byte_array, word_byte_to_word_dict

# define a function to load a pickle
def load_pickle(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as handle:
            de_pickle = pickle.load(handle)
    else:
        de_pickle = None
        print("file does not exist")
    return de_pickle


def byte_encode_words(word:str) -> int:
    ret = 0
    for c in sorted(word):
        alpha_index = ord(c) - ord("a")
        ret |= 1 << alpha_index
    return ret

def build_l2(word_byte_list:list, focal_values:set = None) -> pd.DataFrame:
        
    l2_list = np.full(shape = (1000000, 3), fill_value = -1, dtype = np.int32)
    row_index = 0
    found_values = set()
    for w1_be, w2_be in combinations(word_byte_list, 2):
        if w1_be & w2_be == 0:   
            # they share no letters in common
            l2 = w1_be | w2_be                  

            if focal_values and l2 in focal_values:
                print('here')
                l2_list[row_index, :] = np.array([w1_be, w2_be, l2], dtype = np.int32)
                row_index += 1
            
            if focal_values is None and l2 not in found_values:
                l2_list[row_index, :] = np.array([w1_be, w2_be, l2], dtype = np.int32)
                found_values.add(l2)
                row_index += 1

    # trim the data frame
    l2_list = l2_list[:row_index, :]
    print(l2_list.shape)
    l2_df = pd.DataFrame(data = l2_list, columns = ['w1b', 'w2b', 'l2'])

    return l2_df
    
def build_l3(word_byte_array:np.array, l2_df:pd.DataFrame, focal_values:set = None) -> pd.DataFrame:
    n_columns = 5
    l3_list = np.full(shape = (100000000, n_columns), fill_value = -1, dtype = np.int32)
    start_pos = 0
    
    found_values = set()    
    
    for i_row, row in l2_df.iterrows():    
        w1b, w2b, l2 = row
        # indexer
        positional_idx = (word_byte_array & l2) == 0

        # words with different letters
        output_array_w3b = word_byte_array[positional_idx]    

        # accumulated letters
        output_array_l3 = output_array_w3b | l2

        # get the shape to build the output
        n_pairs = output_array_l3.shape[0]
        temp_array = np.zeros(shape = (n_pairs, n_columns), dtype = np.int32)

        temp_array[:, 0] = w1b
        temp_array[:, 1] = w2b    
        temp_array[:, 2] = output_array_w3b
        temp_array[:, 3] = l2  # (w1b | w2b)
        temp_array[:, 4] = output_array_l3 # (w1b & w2b)    

        if focal_values is None:
            output_array_l3_test = [x not in found_values for x in output_array_l3 ]
            temp_array = temp_array[output_array_l3_test, :]
            found_values.update(temp_array[:, 4])

        n_pairs = temp_array.shape[0]
        end_pos = start_pos + n_pairs
        l3_list[start_pos:end_pos, :] = temp_array
        start_pos = end_pos

        if i_row % 10000 == 0:
            print(i_row)  

    l3_list  = l3_list[:end_pos, :]
    print(l3_list.shape)
    l3_df = pd.DataFrame(data = l3_list, columns = ['w1b', 'w2b', 'w3b', 'l2', 'l3'])

    return l3_df


def build_l4(word_byte_array:np.array, l3_df:pd.DataFrame, focal_values:set=None) -> pd.DataFrame:

    n_columns = 7
    l4_list = np.full(shape = (100000000, n_columns), fill_value = -1, dtype = np.int32)
    start_pos = 0
    
    found_values = set()
        
    for i_row, row in l3_df.iterrows():    
        w1b, w2b, w3b, l2, l3 = row

        # indexer    
        positional_idx = (word_byte_array & l3) == 0
        if positional_idx.size > 0:

            # words with different letters
            output_array_w4b = word_byte_array[positional_idx]    

            # accumulated letters
            output_array_l4 = output_array_w4b | l3

            # get the shape to build the output
            n_pairs = output_array_l4.shape[0]
            temp_array = np.zeros(shape = (n_pairs, n_columns), dtype = np.int32)

            temp_array[:, 0] = w1b
            temp_array[:, 1] = w2b    
            temp_array[:, 2] = w3b
            temp_array[:, 3] = output_array_w4b
            temp_array[:, 4] = l2
            temp_array[:, 5] = l3  # (w1b | w2b | w3b)
            temp_array[:, 6] = output_array_l4 # (w1b | w2b | w3b | w4b) l4
            
            if focal_values is None:
                output_array_l4_test = [x not in found_values for x in output_array_l4 ]
                temp_array = temp_array[output_array_l4_test, :]
                found_values.update(temp_array[:, 6])

            n_pairs = temp_array.shape[0]
            end_pos = start_pos + n_pairs
            l4_list[start_pos:end_pos, :] = temp_array
            start_pos = end_pos

        if i_row % 10000 == 0:
            print(i_row)  

    l4_list = l4_list[:start_pos, :]    
    print(l4_list.shape)
    l4_df = pd.DataFrame(data = l4_list, columns = ['w1b', 'w2b', 'w3b', 'w4b', 'l2', 'l3', 'l4'])

    return l4_df

def build_l5(word_byte_array:np.array, l4_df:pd.DataFrame, focal_values:set = None) -> pd.DataFrame:

    n_columns = 9
    l5_list = np.full(shape = (100000000, n_columns), fill_value = -1, dtype = np.int32)
    start_pos = 0
    
    found_values = set()

    for i_row, row in l4_df.iterrows():    
        w1b, w2b, w3b, w4b, l2, l3, l4 = row

        # indexer    
        positional_idx = (word_byte_array & l4) == 0
        if positional_idx.sum() > 0:

            # words with different letters
            output_array_w5b = word_byte_array[positional_idx]
            #print(positional_idx)
            #print(output_array_w5b)

            # accumulated letters
            output_array_l5 = output_array_w5b | l4

            # get the shape to build the output
            n_pairs = output_array_l5.shape[0]
            temp_array = np.zeros(shape = (n_pairs, n_columns), dtype = np.int32)

            temp_array[:, 0] = w1b
            temp_array[:, 1] = w2b    
            temp_array[:, 2] = w3b
            temp_array[:, 3] = w4b
            temp_array[:, 4] = output_array_w5b
            temp_array[:, 5] = l2  # (w1b | w2b)
            temp_array[:, 6] = l3  # (w1b | w2b | w3b)
            temp_array[:, 7] = l4  # (w1b | w2b | w3b | w4b)
            temp_array[:, 8] = output_array_l5 # (w1b | w2b | w3b | w4b | w5b)            

            if focal_values is None:
                output_array_l5_test = [x not in found_values for x in output_array_l5 ]
                temp_array = temp_array[output_array_l5_test, :]
                found_values.update(temp_array[:, 8])

            n_pairs = temp_array.shape[0]
            end_pos = start_pos + n_pairs
            l5_list[start_pos:end_pos, :] = temp_array
            start_pos = end_pos

        if i_row % 10000 == 0:
            print(i_row)  

    l5_list = l5_list[:start_pos, :]
    print(l5_list.shape)
    l5_df = pd.DataFrame(data = l5_list, columns = ['w1b', 'w2b', 'w3b', 'w4b', 'w5b', 'l2', 'l3', 'l4', 'l5'])

    return l5_df


if __name__ == '__main__':
    print('see five_groups_of_five.ipynb')