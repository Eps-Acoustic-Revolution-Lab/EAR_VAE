import scipy.signal
import torch
import numpy as np


class FIRFilter(torch.nn.Module):
    def __init__(self, filter_type="hp", coef=0.85, fs=44100, ntaps=2049):
        """Initilize FIR pre-emphasis filtering module."""
        super(FIRFilter, self).__init__()
        self.filter_type = filter_type
        self.coef = coef
        self.fs = fs
        self.ntaps = ntaps

        if ntaps % 2 == 0:
            raise ValueError(f"ntaps must be odd (ntaps={ntaps}).")

        if filter_type == "hp":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, -coef, 0]).view(1, 1, -1)
        
        elif filter_type == "fd":
            self.fir = torch.nn.Conv1d(1, 1, kernel_size=3, bias=False, padding=1)
            self.fir.weight.requires_grad = False
            self.fir.weight.data = torch.tensor([1, 0, -coef]).view(1, 1, -1)
        
        elif filter_type == "aw":
            # Definition of analog A-weighting filter according to IEC/CD 1672.
            f1 = 20.598997
            f2 = 107.65265
            f3 = 737.86223
            f4 = 12194.217
            A1000 = 1.9997

            NUMs = [(2 * np.pi * f4) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
            DENs = np.polymul(
                [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
                [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2],
            )
            DENs = np.polymul(
                np.polymul(DENs, [1, 2 * np.pi * f3]), [1, 2 * np.pi * f2]
            )

            self._create_fir_from_analog(NUMs, DENs, fs, ntaps)
        
        elif filter_type == "kw":
            # Stage 1: High-pass filter parameters
            f_hp = 38.135  # Hz
            Q_hp = 0.5
            w_hp = 2 * np.pi * f_hp
            
            # Analog transfer function for the high-pass filter: s^2 / (s^2 + (w/Q)s + w^2)
            NUM_hp = [1, 0, 0]
            DEN_hp = [1, w_hp / Q_hp, w_hp**2]

            # Stage 2: High-shelf filter parameters
            f_shelf = 1681.974 # Hz
            Q_shelf = 0.7
            G_shelf = 4.0 # dB
            k_shelf = 10**(G_shelf / 20.0)
            w_shelf = 2 * np.pi * f_shelf

            # Analog transfer function for the high-shelf filter
            sqrt_k = np.sqrt(k_shelf) # ~1.2589
            NUM_shelf = [k_shelf, (sqrt_k * w_shelf) / Q_shelf, w_shelf**2]
            DEN_shelf = [1, (w_shelf / Q_shelf), w_shelf**2]
            
            # Combine the two filters by multiplying their polynomials
            NUMs = np.polymul(NUM_hp, NUM_shelf)
            DENs = np.polymul(DEN_hp, DEN_shelf)
            
            self._create_fir_from_analog(NUMs, DENs, fs, ntaps)
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")
            
    def _create_fir_from_analog(self, NUMs, DENs, fs, ntaps):
        """Helper function to create FIR filter from an analog prototype."""
        # convert analog filter to digital filter
        b, a = scipy.signal.bilinear(NUMs, DENs, fs=fs)

        # compute the digital filter frequency response
        w_iir, h_iir = scipy.signal.freqz(b, a, worN=16384, fs=fs)

        # then we fit to ntaps FIR filter with least squares
        taps = scipy.signal.firls(ntaps, w_iir, abs(h_iir), fs=fs)

        # now implement this digital FIR filter as a Conv1d layer
        self.fir = torch.nn.Conv1d(
            1, 1, kernel_size=ntaps, bias=False, padding=ntaps // 2
        )
        self.fir.weight.requires_grad = False
        self.fir.weight.data = torch.tensor(taps.astype("float32")).view(1, 1, -1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            input (Tensor): raw signal (B, #channels, #samples).
        Returns:
            Tensor: Filtered signal.
        """
        B, C, T = input.shape
        input = input.view(B*C, 1, T)
        
        input = torch.nn.functional.conv1d(
            input, self.fir.weight.data, padding=self.ntaps // 2
        )
        
        input = input.view(B, C, T)
        
        return input
    
if __name__ == "__main__":

    filter = FIRFilter(filter_type="kw", coef=0.85, fs=16000, ntaps=101)
    x = torch.randn(3, 1, 16000)
    y = filter(x)
    print(y.shape)