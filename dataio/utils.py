import os
import torchaudio
import torch
import pandas as pd
import glob
from random import shuffle

'''
Questions: for multichannel audio, should we sum all channels or keep the tensor
with multichannels?

Current solution: use rms over the channels and squeeze the tensor to 1D.

Thoughts: different channels may contain different info,
i.e. some channel may contain noise (is this possible?)

However, it is possible to preserve channels. Need transpose. [channels, time] -> [time, channels]
'''
def normalize(audio, target_level, EPS = 1e-12):
    '''
    Helper function to normalize audio
    Args:
        audio: torch.tensor, shape = [channels, time]
        target_level: float, target level in dB
    Returns:
        audio: torch.tensor, shape = [time]
    '''
    if not isinstance(audio, torch.Tensor):
        raise TypeError("audio must be a torch.Tensor")
    
    rms = audio.norm(p=2) / (audio.numel() ** 0.5)
    scalar = 10 ** (target_level / 20) / (rms + EPS)
    return audio * scalar

def audioread(path, norm=False, target_level=-25.0):
    '''
    Function to read audio
    Args:
        path: str, path to audio file
        norm: bool, whether to normalize audio
        target_level: float, target level in dB
    Returns:
        audio: torch.tensor, shape = [time]
        sample_rate: int, sample rate
    '''

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError("[{}] does not exist!".format(path))
    try:
        # return tuple(torch.tensor = [channel, time], sample_rate)
        audio, sample_rate = torchaudio.load(path)
    except RuntimeError:
        print('WARNING: Audio type not supported')
        return (None, None)
    
    # rms normalize audio and squeeze mutichannels to 1D: (channels, time) -> (time)
    if len(audio.shape) == 1:
        if norm:
            audio = normalize(audio, target_level)
    else:  # multi-channel
        audio = audio.mean(dim=0)
        if norm:
            audio = normalize(audio, target_level)
    
    return audio.squeeze(), sample_rate

def audiowrite(path, audio, sample_rate=16000, norm=False, target_level=-25.0):
    """
    Function to write audio to disk (WAV or other formats recognized by torchaudio)
    referencing the logic of audioread(). 
    Args:
        path: str
            Path where the audio file will be saved.
        audio: torch.Tensor
            Audio data. May be [time] or [channels, time].
        sample_rate: int
            The sample rate of the audio being saved.
        norm: bool
            Whether to normalize audio before writing (default: False).
        target_level: float
            Target level in dB for normalization (default: -25.0).
    Returns:
        None
    """
    # Optionally normalize:
    if norm:
        audio = normalize(audio, target_level)  # references the same normalize() you provided
    
    # torchaudio.save expects shape = [channels, time], so we add a batch dimension of 1 channel:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)  # shape = [1, time]

    # Ensure directory exists:
    path = os.path.abspath(path)
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # Save audio:
    torchaudio.save(path, audio, sample_rate)

def get_dir(cfg, param_name, assert_exists=False):
    """Helper function to retrieve directory name if it exists,
       create it if it doesn't exist"""

    if param_name in cfg:
        dir_name = cfg[param_name]
    else:
        raise ValueError(f'{param_name} no define in param')
    if assert_exists and not os.path.exists(dir_name):
        raise ValueError(f'{dir_name} does not exist')
    elif not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def extract_list(input_list, index):
    output_list = [i[index] for i in input_list]
    flat_output_list = [item for sublist in output_list for item in sublist]
    flat_output_list = sorted(set(flat_output_list))
    return flat_output_list

def get_file(cfg, param_name):
    """Helper function to retrieve directory name if it exists,
       create it if it doesn't exist"""

    if param_name in cfg:
        file_name = cfg[param_name]
    else:
        raise ValueError(f'{param_name} no define in param')
    dir_name = os.path.dirname(file_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return file_name

def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : tuple
        The file paths passed here should already exist
        in order for the preparation to be considered done.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True

def collect_filenames(csv_path, dir_path, audio_format):
    """
    Collects filenames from a CSV file or directly from a directory.
    If no files are found, raises an error.
    
    Args:
        csv_path (str or None): Path to the CSV file containing filenames (optional).
        dir_path (str): Directory path to search for files if CSV is not provided.
        audio_format (str): Audio file format to search for (e.g., "wav", "mp3").
    
    Returns:
        list: List of filenames.
    
    Raises:
        FileNotFoundError: If no files are found in the directory or if the CSV is empty.
    """
    if csv_path and csv_path != 'None':  # Check if a valid CSV path is provided
        # Load filenames from CSV
        filenames = pd.read_csv(csv_path)['filename'].tolist()
        if len(filenames) == 0:
            raise FileNotFoundError(f"No filenames found in the CSV file: {csv_path}")
    else:  # Search for files in the directory
        # Ensure audio_format does not include wildcards
        if audio_format.startswith("*."):
            audio_format = audio_format[2:]  # Remove leading "*."
        elif audio_format.startswith("."):
            audio_format = audio_format[1:]  # Remove leading "."

        # Generate search pattern for files in the directory
        search_pattern = os.path.join(dir_path, f"*.{audio_format}")
        print(f"Search pattern: {search_pattern}")  # Debug the search pattern
        
        # Collect matching files
        filenames = glob.glob(search_pattern)
        if len(filenames) == 0:
            raise FileNotFoundError(f"No files found in directory: {dir_path} with format: {audio_format}")
    
    # Shuffle the list of filenames
    shuffle(filenames)
    return filenames