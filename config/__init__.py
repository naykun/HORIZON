from utils import *


class BasicArgs:
    # default settings
    resume = True
    use_tqdm = True
    debug = False

    logger.info("Detected System Node %s." % platform.node())
    root_dir = os.getcwd()

    @staticmethod
    def parse_config_name(config_filename):
        """
        Example:
            Args:
                config_filename: 'config/t2i/t2i4ccF8S256.py'
            Return:
                task_name: 't2i'
                model_name: 't2i4ccF8S256'
        """
        task_name, filename = os.path.normpath(config_filename).split(os.path.sep)[-2:]
        model_name = filename.split('.')[0]
        return task_name, model_name
