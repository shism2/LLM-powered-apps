import os
import logging
from datetime import datetime
import pytz


class TZFormatter(logging.Formatter):  
    def converter(self, timestamp):  
        dt = datetime.fromtimestamp(timestamp)  
        tz = pytz.timezone('Etc/GMT-9')  
        return dt.replace(tzinfo=pytz.utc).astimezone(tz)  
  
    def formatTime(self, record, datefmt=None):  
        dt = self.converter(record.created)  
        if datefmt:  
            s = dt.strftime(datefmt)  
        else:  
            s = dt.strftime('%Y-%m-%d %H:%M:%S')  # Format the datetime string without milliseconds  
        return s  

def get_qa_logger(qa_log_folder:str):
    korea_time = datetime.now(pytz.timezone('Asia/Seoul'))  
    year, month, day = korea_time.year, korea_time.month, korea_time.day
    file_name = f"{year}-{month}-{day}_QnA_log.log"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = TZFormatter('%(asctime)s::%(message)s')
    file_hadler = logging.FileHandler( os.path.join(qa_log_folder, file_name), mode='a')
    file_hadler.setFormatter(formatter)
    logger.addHandler(file_hadler)

    return logger

def logging_qa(logger, question, answer):
    logger.info("[Q] " + question)
    logger.info("[A] " + answer)