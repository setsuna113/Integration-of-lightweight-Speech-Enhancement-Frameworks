import random
from dataio.utils import audioread, audiowrite
#from dataio.counter import clean_counter, noise_counter
import os
import torchaudio
import torch

'''
strategy:
does not remove clipped files
'''

# for multiprocessing data initialization
MAXTRIES = 50
MAXFILELEN = 100
clean_counter = 0
noise_counter = 0


def gen_audio(filenum, is_clean, params, length = -1):
    global clean_counter, noise_counter
    """
    Cut audio into correct length from clean or noise files
    Args:
        is_clean: bool, True if clean, False if noise
        params: dict, parameters for constructing audio
        filenum: int, number of file
        length: int, length of audio
    Returns:
        audio: np.array, constructed audio
    """
    print(f"file_{filenum}_{'clean' if is_clean else 'noise'}.wav")
    sample_rate = params['fs']
    silence_dur = params['silence_length']
    if length == -1:
        length = int(params['audio_length'] * sample_rate)
    
    output = torch.zeros(0)
    remaining = length
    files = []

    if is_clean:
        source_files = params['cleanfilenames']
        counter = clean_counter
    else:
        source_files = params['noisefilenames']
        counter = noise_counter

    # Initialize silence using PyTorch
    silence = torch.zeros(int(silence_dur * sample_rate))
    idx = None

    # Iterate until audio is long enough
    while remaining > 0:
        try:
            '''
            with counter.get_lock():
                counter.value += 1
                idx = counter.value % len(source_files)
            '''
            counter += 1
            idx = counter % len(source_files)

            # audioread() should return (audio_tensor, sample_rate)
            input_audio, input_fs = audioread(source_files[idx])

            if input_audio.numel() == 0:
                raise ValueError(f"Empty audio file: {source_files[idx]}")

            # Resample if needed
            if input_fs != sample_rate:
                input_audio = torchaudio.functional.resample(input_audio, input_fs, sample_rate)

            # Subsample if not clean
            if input_audio.shape[-1] > remaining and not is_clean:
                start = random.randint(0, input_audio.shape[-1] - remaining)
                input_audio = input_audio[..., start : start + remaining]

            files.append(source_files[idx])

            # Concatenate audio with output
            output = torch.cat((output, input_audio), dim=-1)
            remaining -= input_audio.shape[-1]\
            
            # Add silence if there's still space
            if remaining > 0:
                silence_len = min(silence.shape[-1], remaining)
                output = torch.cat((output, silence[..., :silence_len]), dim=-1)
                remaining -= silence_len
        
        except Exception as e:
            print(f"CRITICAL ERROR processing {source_files[idx]} (File #{filenum})")
            print(f"Error: {str(e)}")
            print(f"Remaining length: {remaining}")
            raise  # Propagate error to break execution
    # Update global counters identically 
    if is_clean:
        clean_counter = counter
    else:
        noise_counter = counter
    return output, files

def mix_audio(params, filenum):
    """
    Main function for generating audio
    Args:
        params: dict, parameters for constructing audio
    Returns:
        None
    """
    train_list = []
    # generate clean and noise audio
    clean_audio, clean_files = gen_audio(filenum, True, params)
    noise_audio, noise_files = gen_audio(filenum, False, params, len(clean_audio))
    
    # Validate audio dimensions
    for audio in [clean_audio, noise_audio]:
        audio = audio.squeeze()
        if audio.dim() != 1:
            raise ValueError("Audio is not time only")
        audio = audio.unsqueeze(0)
    
    # add noise
    if not params['randomize_snr']:
        snr = params['snr']
    else:
        snr = random.randint(params['snr_lower'], params['snr_upper'])
    clean_snr, noise_snr, noisy_audio, snr = snr_mixer(
        clean=clean_audio,
        noise=noise_audio,
        snr=snr
    )
    # Create unique filenames
    base_name = f"file_{filenum:04d}"
    clean_file = f"{base_name}_clean.wav"
    noise_file = f"{base_name}_noise.wav"
    noisy_file = f"{base_name}_noisy.wav"
    
    # Create full paths
    clean_path = os.path.join(params['clean_proc_dir'], clean_file)
    noise_path = os.path.join(params['noise_proc_dir'], noise_file)
    noisy_path = os.path.join(params['noisyspeech_dir'], noisy_file)
    
    # Save audio files
    audiowrite(clean_path, clean_audio, params['fs'])
    audiowrite(noise_path, noise_audio, params['fs'])
    audiowrite(noisy_path, noisy_audio, params['fs'])
    
    return {
        "clean": clean_path,
        "noise": noise_path,
        "noisy": noisy_path,
        "audio_length": params["audio_length"],
        "snr": snr
    }

def snr_mixer(clean, noise, snr):
    '''Function to mix clean speech and noise at various SNR levels'''
    # Ensure clean and noise are the same length (padding/validation done outside)
    assert clean.shape == noise.shape, "Clean and noise signals must have the same shape."
    # Copy clean waveform to initialize noisy waveform
    noisy_waveform = clean.clone()

    # Compute RMS amplitudes
    clean_rms = torch.sqrt(torch.mean(clean**2) + 1e-14)
    noise_rms = torch.sqrt(torch.mean(noise**2) + 1e-14)

    # Calculate noise scaling factor based on SNR
    noise_amplitude_factor = 1 / (10 ** (snr / 20) + 1)  # Converts SNR dB to linear scale
    new_noise_amplitude = noise_amplitude_factor * clean_rms
    noisy_waveform *= 1 - noise_amplitude_factor
    noise *= new_noise_amplitude / (noise_rms + 1e-14)
    noisy_waveform += noise

    # Return the mixed signals
    return clean, noise, noisy_waveform, snr    




