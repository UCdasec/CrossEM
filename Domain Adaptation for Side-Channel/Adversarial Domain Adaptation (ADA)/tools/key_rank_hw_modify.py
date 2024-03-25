import sys
import pdb
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

import tools.process_data as process_data


# Compute the rank of the real key for a give set of predictions
def rank(predictions, plaintext, real_key, target_byte, HW, hw_mapping):
    # Compute the rank
    key_bytes_proba = np.zeros(256)
    for p in range(0, predictions.shape[0]):
        # Go back from the class to the key byte. '2' is the index of the byte (third byte) of interest.
        plaintext_i = plaintext[p][target_byte]
        for i in range(0, 256):
            # Our candidate key byte probability is the sum of the predictions logs # AES_Sbox[plaintext ^ i]
            tmp_label = HW[process_data.aes_internal(plaintext_i, i)]
            tmp_hw_list = hw_mapping[tmp_label]
            proba = predictions[p][tmp_label]
            if proba != 0:
                proba_log_share = np.log(proba) / len(tmp_hw_list)
                for elem in tmp_hw_list:
                    key_bytes_proba[elem] += proba_log_share
            else:
                # We do not want an -inf here, put a very small epsilon
                # that corresponds to a power of our min non zero proba
                min_proba_predictions = predictions[p][np.array(predictions[p]) != 0]
                if len(min_proba_predictions) == 0:
                    print("Error: got a prediction with only zeroes ... this should not happen!")
                    sys.exit(-1)
                min_proba = min(min_proba_predictions)
                min_proba_log_share = np.log(min_proba**2) / len(tmp_hw_list)
                for elem in tmp_hw_list:
                    key_bytes_proba[elem] += min_proba_log_share

    # Now we find where our real key candidate lies in the estimation.
    # We do this by sorting our estimates and find the rank in the sorted array.
    sorted_proba = np.array(list(map(lambda a: key_bytes_proba[a], key_bytes_proba.argsort()[::-1])))
    real_key_rank = np.where(sorted_proba == key_bytes_proba[real_key])[0][0]
    return real_key_rank


def create_hw_label_mapping():
    ''' this function return a mapping that maps hw label to number per class '''
    HW = defaultdict(list)
    for i in range(0, 256):
        hw_val = process_data.calc_hamming_weight(i)
        HW[hw_val].append(i)
    return HW


def full_ranks(predictions, key, plaintext_attack, target_byte):
    # Real key byte value that we will use. '2' is the index of the byte (third byte) of interest.
    real_key = key[target_byte]

    index = np.arange(1, predictions.shape[0]+1)
    f_ranks = np.zeros((len(index), 2), dtype=np.uint32)

    hw_mapping = process_data.create_hw_label_mapping()
    HW = process_data.get_HW()

    for i in tqdm(range(0, len(index))):
        t = index[i]
        real_key_rank = rank(predictions[:t], plaintext_attack[:t], real_key, target_byte, HW, hw_mapping)
        f_ranks[i] = [t, real_key_rank]
    return f_ranks

