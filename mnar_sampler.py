import numpy as np

def mnar_mask(X: np.ndarray, p: float, mode: str = "magnitude") -> np.ndarray:
    """
    Generate a stochastic binary mask of the same shape as X, where ~p proportion of entries are masked.
    Masking is not uniform random, but biased by data values.

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (n, d), standardized (zero mean, unit variance).
    p : float
        Proportion of entries to mask (0 < p < 1).
    mode : str
        Value-dependence mode:
            - "magnitude": higher |X| entries more likely to be masked
            - "positive": positive entries more likely
            - "negative": negative entries more likely

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape (n, d). 1 = masked, 0 = keep.
    """
    n, d = X.shape
    total_entries = n * d
    k = int(total_entries * p)

    if mode == "magnitude":
        scores = np.abs(X)
    elif mode == "positive":
        scores = np.clip(X, 0, None)
    elif mode == "negative":
        scores = np.clip(-X, 0, None)
    else:
        raise ValueError(f"Unknown mode {mode}")

    # Flatten & normalize scores into probabilities
    scores = scores.ravel()
    if scores.sum() == 0:  # fallback if all scores are zero
        probs = np.full_like(scores, 1/len(scores), dtype=float)
    else:
        probs = scores / scores.sum()

    # Stochastic sampling of exactly k indices
    chosen = np.random.choice(len(scores), size=k, replace=False, p=probs)

    mask = np.ones(len(scores), dtype=int)
    
    #a chosen entry is masked, masked entries are denoted by zero
    mask[chosen] = 0
    
    
    return mask.reshape(n, d)
