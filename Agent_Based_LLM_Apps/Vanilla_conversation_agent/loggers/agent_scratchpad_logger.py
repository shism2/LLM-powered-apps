## Credit: Abubakar Abid (https://github.com/gradio-app/gradio/issues/2362#issuecomment-1424446778)

import logging, sys, os


class ScratchpadLogger:  
    def __init__(self, file_path):  
        self.file_path = file_path
        self.file = open(os.path.join(self.file_path, 'scratch_log.log'), 'w', buffering=1)  
        self.stdout = sys.stdout  
  
    def write(self, data):  
        self.file.write(data)  
        self.stdout.write(data)  
  
    def flush(self):  
        # self.file.flush() 
        pass 


def read_logs_from_file(file_path):
    with open(os.path.join(file_path, 'scratch_log.log'), "r") as f:
        return f.read()