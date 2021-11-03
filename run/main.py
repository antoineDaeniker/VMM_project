import sys
import os
import shutil
import pprint

_root_path = os.path.join(os.path.dirname(__file__) , '..')
sys.path.insert(0, _root_path)
import lib

def main():

    cfg = lib.utils.utils.parse_args().cfg
    trainer = lib.core.trainer.Trainer(cfg)

    ### copy yaml description file to the save folder
    shutil.copy2(
        trainer.cfg.CONFIG_FILENAME,
        trainer.final_output_dir)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)

    trainer.run()


if __name__ == '__main__':
    main()