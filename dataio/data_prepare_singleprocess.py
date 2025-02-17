import os
import glob
from random import shuffle
import pandas as pd
from pathlib import Path
from itertools import repeat
import re
from random import sample
import speechbrain as sb
import json
import csv
from sklearn.utils import resample

from dataio.utils import get_file, get_dir, skip, check_folders, collect_filenames
from dataio.process_audio import mix_audio
#from dataio.counter import clean_counter, noise_counter, init
from dataio.security import validate_audio_settings, validate_dir_settings, compute_file_indexing

PROCESSES = 32

def data_gen(**cfg):
    """
    Main function for multiprocessing
    Args:
        cfg: dict, parameters for constructing audio
    Returns:
        None
    """

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

    train_list = []
    required_fields = ['clean', 'noise', 'noisy', 'audio_length', 'snr']
    
    # 生成原始数据
    for filenum in range(file_settings['num_files']):
        try:
            result = mix_audio(mix_audio_params, filenum)
            
            # 字段完整性检查
            if not all(key in result for key in required_fields):
                raise ValueError(f"Missing fields in {result}")
                
            # 数据类型验证
            if not isinstance(result['audio_length'], (int, float)):
                raise TypeError("audio_length must be numeric")
            if not isinstance(result['snr'], (int, float)):
                raise TypeError("SNR must be numeric")
                
            train_list.append(result)
        except Exception as e:
            print(f"生成文件 {filenum} 失败: {str(e)}")
            continue
    
    # 保存为规范化中间文件（JSON 格式）
    train_list_file = get_file(cfg, 'train_list_file')
    
    with open(train_list_file, 'w') as f:
        json.dump(train_list, f, indent=2)
    
    print(f"中间文件已保存至: {train_list_file}")
    return train_list

def data_prepare(**params):
    if skip(params['train_annotation'], params['valid_annotation'], params['train_annotation']):
        return
    
    if not check_folders(params['noisy_destination'], params['clean_destination'], params['noise_destination']) or \
            not skip(params['train_list_file']):
        data_gen(**params)

    with open(params['train_list_file']) as f:
        train_list = json.load(f)
    
    # 转换为 DataFrame 以便采样（兼容旧逻辑）
    df = pd.DataFrame(train_list)
    total_data = len(df)
    indices = list(range(total_data))
    
    # 数据分割（保持与原始逻辑兼容）
    valid_size = min(int(0.2 * total_data), 2000)
    test_size = min(int(0.1 * total_data), 100)
    
    valid_ids = resample(indices, n_samples=valid_size, replace=False, random_state=42)
    remaining_ids = [i for i in indices if i not in valid_ids]
    test_ids = resample(remaining_ids, n_samples=test_size, replace = False, random_state=42)
    train_ids = [i for i in remaining_ids if i not in test_ids]
    
    # 生成最终数据集
    datasets = {
        'train': {'ids': train_ids, 'path': params['train_annotation']},
        'valid': {'ids': valid_ids, 'path': params['valid_annotation']},
        'test': {'ids': test_ids, 'path': params['test_annotation']}
    }
    
    for key in ['train', 'valid', 'test']:
        dataset_dict = {}
        # 按索引提取数据并结构化
        for idx, data_idx in enumerate(sorted(datasets[key]['ids'])):
            raw_entry = df.iloc[data_idx].to_dict()
            
            entry = {
                #"id": f"{key}_{idx}",
                "clean": raw_entry['clean'],
                "noise": raw_entry['noise'],
                "noisy": raw_entry['noisy'],
                "audio_length": float(raw_entry['audio_length']),
                "snr": int(raw_entry['snr'])
            }
            dataset_dict[f"{key}_{idx}"] = entry
        
        # 保存 JSON
        output_path = Path(datasets[key]['path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        print(f"{key.capitalize()} 数据集 ({len(dataset_dict)} 条) 已保存至: {output_path}")

def dataio_prep(**hparams):
    # 定义音频处理管道（移除频谱生成）
    @sb.utils.data_pipeline.takes("noisy", "clean")
    @sb.utils.data_pipeline.provides("noisy_sig", "clean_sig")  # 移除 spec
    def audio_pipeline(noisy, clean):
        clean_sig = sb.dataio.dataio.read_audio(clean)
        yield clean_sig
        noisy_sig = sb.dataio.dataio.read_audio(noisy)
        yield noisy_sig
    
    # 数据集定义
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }
    
    hparams["dataloader_options"]["shuffle"] = False
    
    for dataset in data_info:
        with open(data_info[dataset]) as f:
            data = json.load(f)
        
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset(
            data=data,
            dynamic_items=[audio_pipeline],
            output_keys=["id", "clean_sig", "noisy_sig"]  # 移除 spec
        ).filtered_sorted(sort_key="audio_length")
        
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