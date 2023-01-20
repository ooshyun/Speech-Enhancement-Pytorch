import numpy as np
from pesq import pesq, cypesq
from pypesq import pesq as nb_pesq
from pystoi import stoi
from museval.metrics import bss_eval

def SDR(reference, estimation, sr=16000):
    """Signal to Distortion Ratio (SDR) from museval

    Reference
    ---------
    - https://github.com/sigsep/sigsep-mus-eval
    - Vincent, E., Gribonval, R., & Fevotte, C. (2006). Performance measurement in blind audio source separation.
    IEEE Transactions on Audio, Speech and Language Processing, 14(4), 1462-1469.

    """
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    sdr_batch = np.empty(shape=(reference_numpy.shape[0], reference_numpy.shape[1]))

    for batch in range(reference_numpy.shape[0]):
        for ch in range(reference_numpy.shape[1]):
            sdr_batch[batch, ch], _, _, _, _ = bss_eval(
                reference_numpy[batch, ch], estimation_numpy[batch, ch]
            )
    sdr_batch = np.mean(sdr_batch)
    return sdr_batch


def SI_SDR(reference, estimation, sr=16000):
    """Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)。

    Args:
        reference: numpy.ndarray, [..., T]
        estimation: numpy.ndarray, [..., T]

    Returns:
        SI-SDR

    References:
        SDR- Half- Baked or Well Done? (http://www.merl.com/publications/docs/TR2019-013.pdf)
    """
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 

    reference_energy = np.sum(reference_numpy**2, axis=-1, keepdims=True)

    optimal_scaling = (
        np.sum(estimation_numpy*reference_numpy, axis=-1, keepdims=True) / (reference_energy + np.finfo(dtype=reference_energy.dtype).eps)
    )

    projection = optimal_scaling * reference_numpy
    noise = estimation_numpy - projection

    ratio = np.sum(projection**2, axis=-1) / (np.sum(noise**2, axis=-1) + np.finfo(dtype=reference_energy.dtype).eps)
    ratio = np.mean(ratio)
    return 10 * np.log10(ratio+np.finfo(dtype=reference_energy.dtype).eps)


def STOI(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    stoi_batch = np.empty(shape=(reference_numpy.shape[0], reference_numpy.shape[1]))
    for batch in range(reference_numpy.shape[0]):
        for ch in range(reference_numpy.shape[1]):
            stoi_batch[batch, ch] = stoi(
                reference_numpy[batch, ch],
                estimation_numpy[batch, ch],
                sr,
                extended=False,
            )

    stoi_batch = np.mean(stoi_batch)
    return stoi_batch


def WB_PESQ(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation        
    num_batch, num_channel = reference_numpy.shape[0], reference_numpy.shape[1]
    pesq_batch = np.empty(shape=(num_batch, num_channel))

    count_error = 0
    for batch in range(num_batch):
        for ch in range(num_channel):
            try:
                pesq_batch[batch, ch] = pesq(
                    sr,
                    reference_numpy[batch, ch],
                    estimation_numpy[batch, ch],
                    mode="wb",
                )
            except cypesq.NoUtterancesError:
                print("cypesq.NoUtterancesError: b'No utterances detected'")
                count_error += 1
    if batch * num_channel - count_error > 0:
        pesq_batch = np.sum(pesq_batch) / (num_batch * num_channel - count_error)
    else:
        pesq_batch = 0
    return pesq_batch


def NB_PESQ(reference, estimation, sr=16000):
    if not isinstance(reference, np.ndarray):
        reference_numpy = reference.numpy()
        estimation_numpy = estimation.numpy()
    else:
        reference_numpy = reference
        estimation_numpy = estimation 
    score = 0

    for ref_batch, est_batch in zip(reference_numpy, estimation_numpy):
        for ref_ch, est_ch in zip(ref_batch, est_batch):
            # print(ref_ch.shape, est_ch.shape, type(ref_ch), type(est_ch))
            # print(ref_ch, est_ch)
            score += nb_pesq(sr, ref_ch, est_ch)
    score /= (
        reference.shape[0] * reference.shape[1]
        if reference.shape[0] is not None
        else reference.shape[1]
    )
    score = np.squeeze(score)
    return score