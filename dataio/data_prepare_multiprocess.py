import os
import glob
from random import shuffle
import pandas as pd
from pathlib import Path
import multiprocessing
from itertools import repeat
import re
from random import sample
import speechbrain as sb
import json

from dataio.utils import get_file, get_dir, skip, check_folders, collect_filenames
from dataio.process_audio import mix_audio
from dataio.counter import clean_counter, noise_counter, init
from dataio.security import validate_audio_settings, validate_dir_settings, compute_file_indexing

PROCESSES = 1

def data_gen(**cfg):
    """
    Main function for multiprocessing
    Args:
        cfg: dict, parameters for constructing audio
    Returns:
        None
    """
    global clean_counter, noise_counter
    clean_counter = 0
    noise_counter = 0
        
    # Audio settings
    audio_settings = {
        'fs': int(cfg['sampling_rate']),
        'audioformat': cfg['audioformat'],
        'audio_length': float(cfg['audio_length']),
        'silence_length': float(cfg['silence_length']),
        'total_hours': float(cfg['total_hours']),
    }

    file_settings = {
        'fileindex_start': int(cfg['fileindex_start']) if cfg['fileindex_start'] != 'None' else None,
        'fileindex_end': int(cfg['fileindex_end']) if cfg['fileindex_end'] != 'None' else None,
    }

    dir_settings = {
        'clean_dir': get_dir(cfg, 'clean_dir', assert_exists=True),
        'noise_dir': get_dir(cfg, 'noise_dir', assert_exists=True),
        'clean_proc_dir': get_dir(cfg, 'clean_destination'),
        'noise_proc_dir': get_dir(cfg, 'noise_destination'),
        'noisyspeech_dir': get_dir(cfg, 'noisy_destination'),
        #'rir_proc_dir': get_dir(cfg, 'rir_destination')
    }

    snr_settings = {
        'is_test_set': bool(cfg['is_test_set']),
        'snr_lower': int(cfg['snr_lower']),
        'snr_upper': int(cfg['snr_upper']),
        'randomize_snr': bool(cfg['randomize_snr']),
        'target_level_lower': int(cfg['target_level_lower']),
        'target_level_upper': int(cfg['target_level_upper']),
    }

    # Validations and updating index
    validate_audio_settings(audio_settings)
    validate_dir_settings(dir_settings)
    compute_file_indexing(file_settings, audio_settings)
    cleanfilenames = collect_filenames(
        csv_path=cfg.get('speech_csv', None),
        dir_path=dir_settings['clean_dir'],
        audio_format=audio_settings['audioformat'],
    )
    print('Clean speech files found:', len(cleanfilenames))
    noisefilenames = collect_filenames(
        csv_path=cfg.get('noise_csv', 'None'),
        dir_path=dir_settings['noise_dir'],
        audio_format=audio_settings['audioformat'],
    )
    print('Noise files found:', len(noisefilenames))


    # Handle RIR settings if applicable
    if cfg['add_reverb_data'] == 1:
        rir_settings = {
            'add_reverb_data': int(cfg['add_reverb_data']),
            'rir_dir': str(cfg['rir_dir']),
            'rir_choice': int(cfg['rir_choice']),
            'lower_t60': float(cfg['lower_t60']),
            'upper_t60': float(cfg['upper_t60']),
            'percent_for_adding_reverb': float(cfg['percent_for_adding_reverb']),
        }
        # Determine the RIR table CSV path
        if audio_settings['fs'] == 16000:
            rir_table_csv = os.path.join(os.path.dirname(__file__), 'datasets/acoustic_params/RIR_table_simple.csv')
        elif audio_settings['fs'] == 48000:
            rir_table_csv = os.path.join(os.path.dirname(__file__), 'datasets/acoustic_params/RIR_table_simple_48k.csv')
        else:
            raise ValueError(f"Unsupported sampling rate: {audio_settings['fs']}")

        rir_settings['myrir'], rir_settings['mychannel'], rir_settings['myt60'] = rir_operations(
            rir_table_csv=rir_table_csv,
            rir_choice=rir_settings['rir_choice'],
            lower_t60=rir_settings['lower_t60'],
            upper_t60=rir_settings['upper_t60'],
        )
    else:
        rir_settings = None
    
    # File indexing logic
    if file_settings['fileindex_start'] is not None and file_settings['fileindex_end'] is not None:
        file_settings['num_files'] = file_settings['fileindex_end'] - file_settings['fileindex_start']
    else:
        file_settings['num_files'] = int(
            (audio_settings['total_hours'] * 60 * 60) / audio_settings['audio_length']
        )
        file_settings['fileindex_start'] = 0
        file_settings['fileindex_end'] = file_settings['num_files']
    print('Number of files to be synthesized:', file_settings['num_files'])

    # Add filenames to parameters for `mix_audio`
    mix_audio_params = {
        'fs': audio_settings['fs'],
        'silence_length': audio_settings['silence_length'],
        'audio_length': audio_settings['audio_length'],
        'cleanfilenames': cleanfilenames,
        'noisefilenames': noisefilenames,
        'randomize_snr': snr_settings['randomize_snr'],
        'snr': cfg.get('snr', None),
        'snr_lower': snr_settings['snr_lower'],
        'snr_upper': snr_settings['snr_upper'],
        'clean_proc_dir': dir_settings['clean_proc_dir'],
        'noise_proc_dir': dir_settings['noise_proc_dir'],
        'noisyspeech_dir': dir_settings['noisyspeech_dir'],
    }

    # multiprocessing
    with multiprocessing.Pool(
        processes=PROCESSES, 
        initializer=init, 
        initargs=(clean_counter, noise_counter)
    ) as pool:
        tasks = zip(
            repeat(mix_audio_params),
            range(file_settings['num_files'])  # File numbers 0,1,2,...
        )
        
        # Correct starmap call with proper arguments
        output_lists = pool.starmap(
            mix_audio,
            tasks  # Proper argument tuples
            # chunksize=...  # Optional: Add if needed (as INTEGER)
        )

    """
    train_lists = [output[2] for output in output_lists]
    flat_train_list = [item for sublist in train_lists for item in sublist]
    flat_train_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))

    # Save the train list to a file
    train_list_file = get_file(cfg, 'train_list_file')
    with open(train_list_file, 'w') as lst:
        for item in flat_train_list:
            lst.write(item + '\n')
    """

    # Sequentially call mix_audio for testing
    #clean_lists, noise_lists, output_lists = mix_audio(mix_audio_params, file_settings['num_files'])

    # Extract data from output_lists and sort by the first number in the "noisy" filename
    output_lists.sort(key=lambda x: int(re.findall(r'\d+', x["noisy"])[0]))

    # Save the train list to a file
    train_list_file = get_file(cfg, 'train_list_file')
    with open(train_list_file, 'w') as lst:
        for entry in output_lists:
            # Write the dictionary values as a space-separated row
            lst.write(f"{entry['clean']} {entry['noise']} {entry['noisy']} {entry['audio_length']} {entry['snr']}\n")
    print(f"Train list saved to: {train_list_file}")

def data_prepare(**params):
    if skip(params['train_annotation'], params['valid_annotation'], params['train_annotation']):
        return
    
    if not check_folders(params['noisy_destination'], params['clean_destination'], params['noise_destination']) or \
            not skip(params['train_list_file']):
        data_gen(**params)

    df = pd.read_csv(
        params['train_list_file'],
        sep='\\s+', 
        header=None, 
        names= ['clean', 'noise', 'noisy', 'audio_length', 'snr']
    )
    df.columns = ['clean', 'noise', 'noisy', 'audio_length', 'snr']
    total_data = df.shape[0]
    df_index = [i for i in range(total_data)]

    # Split the data into training and validation sets
    valid_size = min(int(0.2 * total_data), 2000)
    test_size = min(int(0.1 * total_data), 100)

    valid_ids = sample(df_index, valid_size)
    test_ids = sample([i for i in df_index if i not in valid_ids], test_size)
    train_ids = [i for i in df_index if i not in valid_ids and i not in test_ids]
    
    paths = {
    'train': params["train_annotation"],
    'valid': params["valid_annotation"],
    'test': params["test_annotation"]
    }

    for ids, key in zip([train_ids, valid_ids, test_ids], ['train', 'valid', 'test']):
        ids.sort()
        df_subset = df.iloc[ids]
        Path(os.path.dirname(paths[key])).mkdir(parents=True, exist_ok=True)
        suffix = Path(paths[key]).suffix

        if suffix == '.csv':
            df_subset.to_csv(paths[key], index=False)
            print(f'{key} set saved to {paths[key]}')
        elif suffix == '.json' or suffix == '':
            records = df_subset.to_dict(orient='records')
            for idx, record in enumerate(records):
                record['id'] = f"{key}_{idx}"
            with open(paths[key], 'w') as f:
                for record in records:
                    f.write(json.dumps(record) + '\n')
            print(f'{key} set saved to {paths[key]}')
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

def dataio_prep(**hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.

    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json` and `valid.json` manifest files are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("noisy_wav", "clean_wav")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig", "spec", "clean_spec")
    def audio_pipeline(noisy_wav, clean_wav):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`.
        """
        clean_sig = sb.dataio.dataio.read_audio(clean_wav)
        yield clean_sig
        noisy_sig = sb.dataio.dataio.read_audio(noisy_wav)
        yield noisy_sig
        #clean_spec = hparams["compute_STFT"](clean_sig)
        #yield clean_spec
        #spec = hparams["compute_STFT"](noisy_sig)
        #yield spec
    
    # Define datasets sorted by ascending lengths for efficiency
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    hparams["dataloader_options"]["shuffle"] = False
    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            dynamic_items=[audio_pipeline],
            output_keys=["id", "clean_sig", "noisy_sig"],
        ).filtered_sorted(sort_key="length")
    return datasets

def rir_operations(rir_table_csv, rir_choice, lower_t60, upper_t60):
    """
    Handles RIR-related operations, including filtering and selecting RIRs.
    
    Args:
        rir_table_csv (str): Path to the RIR table CSV file.
        rir_choice (int): Choice of RIR type (1: real, 2: synthetic, 3: both).
        lower_t60 (float): Lower bound for T60 filtering.
        upper_t60 (float): Upper bound for T60 filtering.
    
    Returns:
        tuple: (myrir, mychannel, myt60), lists of RIR paths, channels, and T60 values.
    """
    # Read RIR table
    temp = pd.read_csv(rir_table_csv, 
        skiprows=[1], 
        sep=',', 
        header=None,
        names=['wavfile', 'channel', 'T60_WB', 'C50_WB', 'isRealRIR']
    )
    
    rir_wav = temp['wavfile'][1:]
    rir_channel = temp['channel'][1:]
    rir_t60 = temp['T60_WB'][1:]
    rir_isreal = temp['isRealRIR'][1:]

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2 = [w for w in rir_isreal]

    # Initialize outputs
    myrir = []
    mychannel = []
    myt60 = []

    # Filter RIRs based on choice
    if rir_choice == 1:  # Real RIRs
        real_indices = [i for i, x in enumerate(rir_isreal2) if x == "1"]
        chosen_i = [i for i in real_indices if lower_t60 <= float(rir_t60_2[i]) <= upper_t60]
    elif rir_choice == 2:  # Synthetic RIRs
        synthetic_indices = [i for i, x in enumerate(rir_isreal2) if x == "0"]
        chosen_i = [i for i in synthetic_indices if lower_t60 <= float(rir_t60_2[i]) <= upper_t60]
    elif rir_choice == 3:  # Both Real and Synthetic
        all_indices = range(len(rir_isreal2))
        chosen_i = [i for i in all_indices if lower_t60 <= float(rir_t60_2[i]) <= upper_t60]
    else:  # Default (Both Real and Synthetic)
        all_indices = range(len(rir_isreal2))
        chosen_i = [i for i in all_indices if lower_t60 <= float(rir_t60_2[i]) <= upper_t60]

    # Populate RIR outputs
    myrir = [rir_wav2[i] for i in chosen_i]
    mychannel = [rir_channel2[i] for i in chosen_i]
    myt60 = [rir_t60_2[i] for i in chosen_i]

    return myrir, mychannel, myt60