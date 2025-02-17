import os


def validate_dir_settings(dir_settings):
    """
    Validates the directory settings by checking if the specified directories exist.
    
    Args:
        dir_settings (dict): Dictionary containing directory paths.
    
    Raises:
        FileNotFoundError: If any required directory does not exist.
    """
    required_dirs = ['clean_dir', 'noise_dir', 'clean_proc_dir', 'noise_proc_dir', 'noisyspeech_dir']
    for dir_key in required_dirs:
        dir_path = dir_settings[dir_key]
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Required directory does not exist: {dir_path}")


def validate_audio_settings(audio_settings):
    """
    Validates the audio settings.
    
    Args:
        audio_settings (dict): Dictionary containing audio-related settings.
    
    Raises:
        ValueError: If any of the audio settings are invalid.
    """
    # Validate sampling rate
    if audio_settings['fs'] not in [16000, 48000]:
        raise ValueError(f"Unsupported sampling rate: {audio_settings['fs']}. Supported rates: 16000, 48000.")

    # Validate audio length, silence length, and total hours
    if audio_settings['audio_length'] <= 0:
        raise ValueError("Audio length must be greater than 0.")
    if audio_settings['silence_length'] < 0:
        raise ValueError("Silence length cannot be negative.")
    if audio_settings['total_hours'] <= 0:
        raise ValueError("Total hours must be greater than 0.")
    
def compute_file_indexing(file_settings, audio_settings):
    """
    Computes the file indexing logic for data generation.
    
    Args:
        file_settings (dict): Dictionary containing file indexing settings.
            - 'fileindex_start': Start index (optional).
            - 'fileindex_end': End index (optional).
        audio_settings (dict): Dictionary containing audio-related settings.
            - 'total_hours': Total duration of audio in hours.
            - 'audio_length': Duration of each audio file.
    
    Returns:
        None: Updates the `file_settings` dictionary in-place.
    """
    if file_settings['fileindex_start'] is not None and file_settings['fileindex_end'] is not None:
        file_settings['num_files'] = file_settings['fileindex_end'] - file_settings['fileindex_start']
    else:
        file_settings['num_files'] = int(
            (audio_settings['total_hours'] * 60 * 60) / audio_settings['audio_length']
        )
        file_settings['fileindex_start'] = 0
        file_settings['fileindex_end'] = file_settings['num_files']
    print('Number of files to be synthesized:', file_settings['num_files'])

