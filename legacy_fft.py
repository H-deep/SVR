# legacy_fft.py — compatibility for old torch.rfft/irfft calls on PyTorch 1.8+
import torch

def _norm_name(normalized: bool):
    # old API's normalized=True matched 'ortho', otherwise 'backward'
    return 'ortho' if normalized else 'backward'

def rfft(x, signal_ndim=1, normalized=False, onesided=True):
    """
    Mimic deprecated torch.rfft:
    - returns real/imag in an extra last dim of size 2
    - supports 1D/2D/3D via signal_ndim
    """
    if x.is_complex():
        raise TypeError("legacy rfft expects real input")
    dims = tuple(range(-signal_ndim, 0))
    y = torch.fft.rfftn(x, dim=dims, norm=_norm_name(normalized))
    # Old rfft returned real/imag stacked in last dim=2
    return torch.view_as_real(y)

def irfft(x, signal_ndim=1, normalized=False, onesided=True, signal_sizes=None):
    """
    Mimic deprecated torch.irfft:
    - expects input with last dim=2 (real/imag)
    - 'signal_sizes' maps to 's' in irfftn
    """
    if x.size(-1) != 2:
        raise ValueError("legacy irfft expects last dim size 2 (real/imag)")
    dims = tuple(range(-signal_ndim, 0))
    y = torch.view_as_complex(x)
    out = torch.fft.irfftn(y, s=signal_sizes, dim=dims, norm=_norm_name(normalized))
    return out
