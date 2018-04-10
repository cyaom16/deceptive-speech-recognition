import numpy as np


def load_data(data_dir, suffix='_'):
    data = np.load(data_dir)
    return data['X' + suffix], data['y' + suffix], data['l' + suffix]


def make_batches(*arrays, batch_size=1, shuffle=True):
    """
    Batch maker

    Args:
        array_1, array_2, ... : array_like
        batch_size (int): batch size
        shuffle (bool, optional): True for shuffle, False otherwise

    Returns:
        A list of array batches

    """
    arrays = list(arrays)
    num_samples = arrays[0].shape[0]

    if shuffle:
        s = np.arange(num_samples)
        np.random.shuffle(s)
        arrays = [a[s] for a in arrays]

    return [np.array_split(a, num_samples // batch_size) for a in arrays]


def resample(inputs, seq_lens, tgt_len):
    """
    Splice input sequences into sub-sequences with a given sample length

    Args:
        inputs (NumPy array):   Raw variable-length sequences
        seq_lens (NumPy array): Sequence lengths
        tgt_len (int):          Target sequence length

    Returns:
        resampled (list): Variable-length sub-sequences

    """
    threshold_index = seq_lens > tgt_len
    resampled = []
    for i, x in enumerate(inputs):
        if threshold_index[i]:
            resampled += np.array_split(x, seq_lens[i] // tgt_len)
        else:
            resampled.append(x)
    return resampled


def pad_sequences(inputs, seq_lens, max_len=None):
    """
    Performs zero-padding on the given sequences to the maximum length

    Args:
        inputs (NumPy array):    Raw variable-length sequences
        seq_lens (NumPy array):  Sequence lengths
        max_len (int, optional): Max sequence length

    Returns:
        inputs (NumPy array): Padded sequences

    """
    max_len = max(seq_lens) if max_len is None else max_len
    for i, x in enumerate(inputs):
        inputs[i] = np.pad(x, ((0, max_len - seq_lens[i]), (0, 0)), 'constant')
    return inputs
