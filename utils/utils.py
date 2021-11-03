import logging

from pathlib import Path
import argparse


def init_exp_folder(output_dir, exp_foldername):
    assert exp_foldername is not None, 'Specify the Experiment Name!'
    assert output_dir is not None, 'Specify the Output folder (where to save models, checkpoints, etc.)!'
    root_output_dir = Path(output_dir)

    # set up logger
    if not root_output_dir.exists():
        print(f'=> creating "{root_output_dir}" ...')
        root_output_dir.mkdir()
    else:
        print(f'Folder "{root_output_dir}" already exists.')

    final_output_dir = root_output_dir / exp_foldername

    if not final_output_dir.exists():
        print('=> creating "{}" ...'.format(final_output_dir))
    else:
        print(f'Folder "{final_output_dir}" already exists.')

    final_output_dir.mkdir(parents=True, exist_ok=True)
    return final_output_dir

def generate_new_name_for_log(log_name):
    log_name = log_name[:-4] # remove ".log"


def create_logger(output_dir, exp_foldername):
    assert exp_foldername is not None, 'Specify the Experiment Name!'
    assert output_dir is not None, 'Specify the Output folder (where to save models, checkpoints, etc.)!'

    log_filename = f'{exp_foldername}'

    i = 1
    while True:
        if (Path(output_dir) / Path(f'{log_filename}_{i:03d}.log')).exists():
            i += 1
        else:
            log_filename = f'{log_filename}_{i:03d}'
            break
    
    print(f'=> New log file "{log_filename}.log" is created.')

    final_log_file = Path(output_dir) / Path(log_filename+'.log') 
    logging.basicConfig(
                        filename=str(final_log_file),# level=logging.INFO,
                        format='%(asctime)-15s %(message)s', 
                        datefmt='%d-%m-%Y, %H:%M:%S'
                        )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(description='Training Launch')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/exp.yaml',
                        required=False,
                        type=str)
    args, rest = parser.parse_known_args()
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    output_dir = 'TESTFOLDER'
    exp_name = 'EXPTESTNAME'
    final_output_dir = init_exp_folder(output_dir, exp_name)
    logger = create_logger(final_output_dir, exp_name)