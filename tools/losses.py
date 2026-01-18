from filter import FIRFilter
import torch


def wrap_to_pi(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def stft_torch(x, fft_size, hop_size, win_length, window, normalized=False):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
        Tensor: Phase spectrogram (B, #frames, fft_size // 2 + 1).
        Tensor: STFT complex spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x,
                        fft_size,
                        hop_size,
                        win_length,
                        window,
                        center=False, 
                        pad_mode=None,
                        normalized=normalized,
                        return_complex=True)
    
    magnitude = torch.abs(x_stft)
    # magnitude = normalize_mag(magnitude)
    
    phase = torch.angle(x_stft)
    phase = wrap_to_pi(phase)
    
    return magnitude.transpose(2, 1), phase.transpose(2, 1), x_stft.transpose(2, 1)


class PhaseLoss_GroupDelay(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicts_phase, targets_phase):
        predicts_if = torch.diff(predicts_phase, dim=-1)
        targets_if = torch.diff(targets_phase, dim=-1)
        return torch.nn.functional.l1_loss(predicts_if, targets_if)
    

class PhaseLoss_InstantaneousFrequency(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicts_phase, targets_phase):
        predicts_if = torch.diff(predicts_phase, dim=1)
        targets_if = torch.diff(targets_phase, dim=1)
        
        return torch.nn.functional.l1_loss(predicts_if, targets_if)


class PhaseCorrelationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, predicts_stft, targets_stft):
        """
        Args:
            predicts_stft: [B, T, fft_size // 2 + 1]
            targets_stft: [B, T, fft_size // 2 + 1]

        """
        predicts_stft = predicts_stft.transpose(2, 1) # [B, fft_size // 2 + 1, T]
        targets_stft = targets_stft.transpose(2, 1) # [B, fft_size // 2 + 1, T]
        
        R = predicts_stft * torch.conj(targets_stft)
        R_norm = R / (torch.abs(predicts_stft) * torch.abs(targets_stft) + 1e-8)
        
        phase_corr = torch.real(R_norm).mean(dim=1) # [B, T]
        
        return 1-phase_corr.mean() # [B]


if __name__ == "__main__":
    mock_audio = torch.randn(3, 16000)
    window = torch.hann_window(512)
    mag, phase, stft = stft_torch(mock_audio, 512, 128, 512, window)

    phase_loss_gd = PhaseLoss_GroupDelay()
    phase_loss_if = PhaseLoss_InstantaneousFrequency()
    phase_corr_loss = PhaseCorrelationLoss()

    loss_gd = phase_loss_gd(phase, phase)
    loss_if = phase_loss_if(phase, phase)
    loss_corr = phase_corr_loss(stft, stft)

    print(loss_gd.item(), loss_if.item(), loss_corr.item())
