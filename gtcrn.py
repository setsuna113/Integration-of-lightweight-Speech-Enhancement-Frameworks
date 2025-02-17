"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


class STFT_Layer(nn.Module):
    def __init__(self, frame_len, frame_hop):
        super(STFT_Layer, self).__init__()
        self.eps = torch.finfo(torch.float32).eps
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, x):
        """
        Forward pass for STFT computation.
        
        Args:
        -----
        x: torch.Tensor
            Input time-domain signal, shape (B, T).

        Returns:
        --------
        spec: torch.Tensor
            STFT spectrogram, shape (B, F, T, 2).
        """
        if len(x.shape) != 2:
            raise ValueError("Input x must be of shape [B, T], where B is batch size and T is time.")
        
        # Compute STFT; return_complex=True gives a complex tensor
        y = torch.stft(
            x,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            return_complex=True,
            center=True,  # Centered STFT
        )
        # Separate real and imaginary parts
        spec = torch.view_as_real(y)  # Shape: (B, F, T, 2)
        return spec


class ISTFT_Layer(nn.Module):
    def __init__(self, frame_len, frame_hop):
        super(ISTFT_Layer, self).__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

    def forward(self, spec):
        """
        Forward pass for ISTFT computation.
        
        Args:
        -----
        spec: torch.Tensor
            STFT spectrogram, shape (B, F, T, 2).

        Returns:
        --------
        waveform: torch.Tensor
            Reconstructed time-domain signal, shape (B, T).
        """
        # Convert real + imaginary parts back to a complex tensor
        spec = spec.contiguous()
        y = torch.view_as_complex(spec)  # Shape: (B, F, T)
        
        # Compute ISTFT
        waveform = torch.istft(
            y,
            n_fft=self.frame_len,
            hop_length=self.frame_hop,
            win_length=self.frame_len,
            center=True,  # Centered ISTFT
        )
        return waveform


class ERB(nn.Module):
    """
    The ERB class implements a forward and inverse transformation between 
    frequency-domain spectrograms and ERB subbands using triangular filter banks.   
    Key Features:
    - Converts frequency bins to a reduced ERB subband representation.
    - Can reconstruct (approximately) the original frequency bins using inverse ERB.
    - Equivalent Rectangular Bandwidth (ERB) scale.

    Arguments
    ---------
    erb_subband_1 (int): 
        Index of the first frequency bin to use (low cutoff frequency).
    erb_subband_2 (int): 
        Number of ERB subbands (resolution of the ERB representation).
    nfft (int): 
        Number of FFT points (default: 512).
    high_lim (float): 
        Upper frequency limit for the ERB filters (in Hz, default: 8000 Hz).
    fs (float): 
        Sampling rate of the signal (in Hz, default: 16000 Hz).
    """

    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        # Generate the ERB filter bank matrix (fixed)
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        
        # Total number of frequency bins
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        # Input size: (nfreqs - erb_subband_1); Output size: erb_subband_2
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        # Input size: erb_subband_2; Output size: (nfreqs - erb_subband_1)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        
        # Set the weights (fixed)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        # Inverse ERB is simply transpoed matrix of filters
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        """
        Internal function that converts a frequency (in Hz) to the ERB scale.
        Gets called by erb_filter_banks.
        """
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        """
        Internal function that converts a frequency (in ERB scale) to Hz.
        """
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz
    
    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        """
        Generates the ERB filter bank matrix with triangular filters.

        Args:
            erb_subband_1 (int): Starting frequency bin (low cutoff).
            erb_subband_2 (int): Number of ERB filters (resolution).
            nfft (int): Number of FFT points.
            high_lim (float): Upper frequency limit (in Hz).
            fs (float): Sampling rate (in Hz).

        Returns:
            torch.Tensor: ERB filter bank matrix of size (erb_subband_2, nfft//2 + 1 - erb_subband_1).
        """
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        # 1e-12 to prevent zero division
        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        # triangular filter with certral frequency (weight 1) at bins[i+1]
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """
        Converts a frequency-domain spectrogram to ERB subbands.
        
        Args:
            x (torch.Tensor): Input spectrogram of shape (B, C, T, F),
                            where F = number of frequency bins.
        
        Returns:
            torch.Tensor: ERB subband representation of shape (B, C, T, F_erb),
                        where F_erb = erb_subband_1 + erb_subband_2.
        """
        # extract and apply ERB transformation to high frequency bins
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])

        # Concatenate the low-frequency bins
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """
        Converts ERB subbands back to a frequency-domain spectrogram.
        
        Args:
            x_erb (torch.Tensor): ERB subband representation of shape (B, C, T, F_erb).
        
        Returns:
            torch.Tensor: Reconstructed spectrogram of shape (B, C, T, F).
        """
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction, applies a sliding window (via nn.Unfold) along the frequency 
    dimension to capture local frequency context for each time frame.
    
    Arguments
    ---------
    kernel_size (int): 
        The size of the window along the frequency dimension
        (number of frequency bins to include).
    stride (int):
        The step size for the sliding window along the frequency dimension.

    Example
    -------
    >>> sfe = SFE(kernel_size=3, stride=1)
    >>> x = torch.randn(2, 4, 10, 20)  # (B=2, C=4, T=10, F=20)
    >>> xs = sfe(x)
    >>> print(xs.shape)
    """
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        # padding avoids truncating the frequency dimension at edge by adding 0 to 
        # the beginning and end of F but not along the time dimension
        # (B, C, T, F) -> (B, C, T, F+2*padding)

        # unfold the frequncy into another dimension because pytorch don't accpect vector as element
        # (B, C, T, F) -> (B, C, T, (kernel_size*F)) -> (B, kernel_size*C, kernel_size*T, kernel_size)
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """
        Arguments
        ---------
        x (torch.Tensor):
            Input tensor of shape (B, C, T, F), where:
            - B: Batch size.
            - C: Number of channels.
            - T: Number of time frames.
            - F: Number of frequency bins (or ERB subbands).

        Returns
        -------
        xs (torch.Tensor): 
            Output tensor of shape (B, C * kernel_size, T, F),
            resulting in `C * kernel_size` channels.
        """
        # reshape 
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention applies a temporal attention mechanism to the input tensor `x` 
    using an attention vector generated by a GRU-based recurrent network.
    Determines the importance of each time frame in the input tensor `x` based on the 
    frequency content, and weights all over the frequency spectrum for each time frame.
    
    Arguments
    ---------
    channels (int):
        Number of channels in the input tensor `x`.
    """
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """
        Arguments
        ---------
        x: (B,C,T,F) pytorch tensor
        
        Returns
        -------
        x: (B,C,T,F) pytorch tensor
        """
        # zt shape: (B, C, T) → This summarizes the frequency content per time frame
        zt = torch.mean(x.pow(2), dim=-1)
        # GRU processes the transposed zt (B, T, C) -> (B, T, 2*C)
        at = self.att_gru(zt.transpose(1,2))[0]
        # linear layer to reduce dimention (B, T, 2*C) -> (B, c, T)
        at = self.att_fc(at).transpose(1,2)
        # sigmoid activation to generate the attention weights
        at = self.att_act(at)
        # expand the attention weights to the same shape as x -> (B,C,T,1)
        At = at[..., None]

        # apply the attention weights to the input tensor x
        return x * At


class ConvBlock(nn.Module):
    """
    Convolution, nomorlization and activation block.
    Arguments
    ---------
    in_channels (int):
        Number of input channels.
    out_channels (int):
        Number of output channels.
    kernel_size (tuple):
        Size of the convolutional kernel.
    stride (int or tuple):
        Stride of the convolutional kernel.
    padding (int or tuple):
        Padding of the convolutional kernel.
    groups (int):
        Number of groups in the convolutional layer.
    use_deconv (bool):
        Whether to use a transposed convolution.
    is_last (bool):
        Whether this is the last layer in the network.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        # PReLU activation for all layers except the last one
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        """
        Applies the convolution, normalization, and activation operations to the 
        input tensor `x`.
        
        Arguments
        ---------
        x (torch.Tensor): (B, C, T, F)

        Returns
        -------
        x (torch.Tensor): (B, C_out, T_out, F_out)
        """
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """
    Group Temporal Convolution: 
        SFE: extract local frequency features
        Pointwise Convolution: reduce number of channels
        Depthwise Convolution: spatial convolution independently for each channel
            Captures spatial/temporal patterns within each channel.
        Pointwise Convolution: Maps to the desired number of output channels.
        TRA: weights the time frames.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)
        
        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x =  self.shuffle(h1, x2)
        
        return x


class GRNN(nn.Module):
    """Grouped RNN:
    This module splits the input into two groups along the feature dimension,
    processes each group independently using separate GRUs, and concatenates the outputs.

    Arguments
    ---------
    input_size (int):
        Number of input features dimension.
    hidden_size (int):
        Size of the hidden state for each GRU.
    num_layers (int):
        Number of recurrent layers per GRU(default: 1).
    batch_first (bool):
        If True, input/output tensors are of shape (B, seq_length, input_size) (default: True).
    bidirectional (bool):
        If True, the GRU is bidirectional (default: False).
    """
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        Arguments
        ---------
        x (torch.Tensor): 
            Input tensor of shape (B, seq_length, input_size)
        h: (torch.Tensor, optional):
            Hidden state tensor of shape (num_layers, B, hidden_size).
            If None, it will be initialized to zeros.
        
        Returns
        -------
        y (torch.Tensor):
            Output tensor of shape (B, seq_length, hidden_size)
        h (torch.Tensor):
            Hidden state tensor of shape (num_layers, B, hidden
        """
        # Initialize hidden state
        if h == None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        
        # Split the input tensor along the feature dimension -> (B, seq_length, input_size//2)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        # Split hidden state into two groups for GNU -> (num_layers, B, hidden_size//2)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        # Forward pass through the GRUs
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        # Concatenate the outputs of 2 GNU along the feature dimension
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN: 
    Processes input along both the time axis (intra-path) and the frequency axis (inter-path)
    using grouped RNNs, followed by residual connections and normalization.

    Arguments
    ---------
    input_size (int):
        Size of input features dimension (C).
    width (int):
        Width of the feature map (F).
    hidden_size (int):
        Size of the hidden state for each RNN.
    """
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        # Intra-path RNN (along time axis, T), bidirectional
        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        # Inter-path RNN (along frequency axis, F), unidirectional
        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """Forward pass of DPGRNN.
        Arguments
        ---------
        x (torch.Tensor):
            Input tensor of shape (B, C, T, F).
        
        Returns
        -------
        dual_out (torch.Tensor):
            Output tensor of shape (B, C, T, F).
        """
        # Intra RNN reshape (B, C, T, F)-> (B*T,F,C)
        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]  # output only, no hidden state
        intra_x = self.intra_fc(intra_x)
        # (B*T,F,C) -> (B,T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x) # normalize
        # Residual connection
        intra_out = torch.add(x, intra_x)

        # Inter RNN reshape (B, T, F, C) -> (B*F,T,C)
        x = intra_out.permute(0,2,1,3) # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0] # output only, no hidden state
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) # normalize
        # Residual connection
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*5,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*1,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    def __init__(self, frame_len=512, frame_hop=128, window='hann'):
        super().__init__()
        self.frame_len = frame_len
        self.frame_hop = frame_hop

        # STFT and ISTFT layers
        self.stft = STFT_Layer(frame_len, frame_hop)
        self.istft = ISTFT_Layer(frame_len, frame_hop)
        
        # (B, 3, T, 257) -> (B, 3, T, 129)
        # 3 channels: magnitude, real, imaginary
        self.erb = ERB(65, 64)
        # (B, 3, T, 129) -> (B, 9, T, 129)
        self.sfe = SFE(3, 1)

        # Compress the frequency features: (B, 9, T, 129) -> (B, 16, T, 33)
        self.encoder = Encoder()
        
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        
        # Reconstruct the frequency features: (B, 16, T, 33) -> (B, 9, T, 129)
        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, x):
        """
        spec: (B, F, T, 2); 2 channels: real and imaginary
        """
        # Compute STFT
        spec = self.stft(x)  # (B, F, T, 2)        
        spec_ref = spec  # (B,F,T,2)

        # 3 * (B, T, F) -> (B, 3, T, F) by stacking
        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs = self.encoder(feat)
        
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec_ref.permute(0,3,2,1)) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        enhanced_waveform = self.istft(spec_enh)  # (B, T)
        
        return enhanced_waveform


'''
# Instantiate the model
model = GTCRN(frame_len=512, frame_hop=128)
# Example input waveform (batch of raw signals)
waveform = torch.randn(4, 16000)  # Batch of 4 signals, 1 second each at 16 kHz
# Forward pass through the model
enhanced_waveform = model(waveform)
print("Enhanced waveform shape:", enhanced_waveform.shape) 
'''
'''
if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (257, 63, 2), as_strings=True,
                                        print_per_layer_stat=True, verbose=True)
    print(flops, params)
    

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)
    
    x1 = torch.stft(x1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    x2 = torch.stft(x2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y1 = model(x1)[0]
    y2 = model(x2)[0]
    y1 = torch.istft(y1, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    y2 = torch.istft(y2, 512, 256, 512, torch.hann_window(512).pow(0.5), return_complex=False)
    
    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
'''