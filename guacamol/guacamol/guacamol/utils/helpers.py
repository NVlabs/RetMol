# ---------------------------------------------------------------
# Taken from the following link as is from:
# https://github.com/BenevolentAI/guacamol/blob/master/guacamol/utils/helpers.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_GUACAMOL).
# ---------------------------------------------------------------

import logging


def setup_default_logger():
    """
    Call this function in your main function to initialize a basic logger.

    To have more control on the format or level, call `logging.basicConfig()` directly instead.

    If you don't initialize any logger, log entries from the guacamol package will not appear anywhere.
    """
    logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
