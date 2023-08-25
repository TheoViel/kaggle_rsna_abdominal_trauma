import numpy as np


def rle_encode(x, fg_val=1):
    """
    Run-Length Encode (RLE) the binary mask `x`.

    Args:
        x (numpy array): Binary mask to be encoded.
        fg_val (int, optional): The foreground value in the mask. Defaults to 1.

    Returns:
        list: A list representing the run-length encoding of the binary mask.
    """
    dots = np.where(x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Converts a list to its string representation.

    Args:
        x (list): The input list to be converted.

    Returns:
        str: A string representation of the input list. If the list is empty, returns '-'.
    """
    if x:  # non-empty list
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s
