# ----------------------------------------------------------------------------------------------------------------------
# Overview: THIS MODULE ADDS LOGGING FUNCTIONALITY INTO ANY MODULE INTO WHICH IT IS LOADED
# @author: ANISH AGARWAL
# ----------------------------------------------------------------------------------------------------------------------


# GLOBAL IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
import logging


# LOGGER SETUP
# ----------------------------------------------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)


# INSTRUCTIONS
# ----------------------------------------------------------------------------------------------------------------------

# Import this file to get logging functionality into your script

# YOU MUST INITIALIZE THE LOG LEVEL (DEBUG,INFO,WARNING,ERROR,CRITICAL) IN THE BEGINNING
# DEBUG is the lowest log and CRITICAL is the highest log level
# Setting the log to a certain level means that any logs higher then that level will also be logged

# BELOW ARE EXAMPLES OF HOW TO SET THE LOGGER LEVEL (COPY AND PASTE ONE OF THESE AT THE BEGINNING OF YOUR PYTHON FILE)
# logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)
# logger.setLevel(logging.CRITICAL)

# BELOW ARE EXAMPLES OF HOW TO PRINT A LOG (paste this into your python file and change the log message)
# logger.debug('Enter the debug log message here')
# logger.info('Enter the info log message here')
# logger.warning('Enter the warning log message here')
# logger.error('Enter the error log message here')
# logger.critical('Enter the critical log message here')
