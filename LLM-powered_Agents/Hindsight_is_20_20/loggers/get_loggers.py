import os
import logging
from datetime import datetime
import pytz

# DEBUG: Detailed information, typically of interest only when diagnosing problems.

# INFO: Confirmation that things are working as expected.

# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g. ‘disk space low’). The software is still working as expected.

# ERROR: Due to a more serious problem, the software has not been able to perform some function.

# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.


class CustomTZFormatter(logging.Formatter):  
    def converter(self, timestamp):  
        dt = datetime.fromtimestamp(timestamp)  
        tz = pytz.timezone('Etc/GMT-9')  
        return dt.replace(tzinfo=pytz.utc).astimezone(tz)  
  
    def formatTime(self, record, datefmt=None):  
        dt = self.converter(record.created)  
        return dt.strftime(datefmt) if datefmt else dt.strftime('%Y-%m-%d %H:%M:%S')



def get_hd22_file_logger(log_file: str, logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = CustomTZFormatter('[%(asctime)s:%(levelname)s] %(message)s')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_hd22_stream_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    formatter = CustomTZFormatter('[%(levelname)s] %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger