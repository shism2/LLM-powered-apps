import os
from typing import Literal

def save_template_as_txt(template:str, prompt_type:Literal['system', 'human', 'ai'], folder:str= '.'):
    with open(os.path.join(folder, f'{prompt_type}_msg.txt'), 'w') as f:
        f.write(template)    
