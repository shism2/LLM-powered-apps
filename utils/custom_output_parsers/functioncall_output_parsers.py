import json
from pydantic import Field
from langchain.schema import AIMessage

class ReActOutputJsonKeyStrParser():


    def __init__(self, key: str):
        self.key = key

    def __call__(self, response: AIMessage) -> str:
        """ 
        response : AIMessage with 'content='' being empty and 'additional_kwargs' being not empty
        output : concatenated string of the values of response.additional_kwargs['function_call']['arguments']
        """
        dict_blob = json.loads(str(response.additional_kwargs['function_call']['arguments']))
        target_dict_blob = dict_blob[self.key] 
        output = '\n'.join(f"{i+1}.  {list(item.values())[0]}" for i, item in enumerate(target_dict_blob)) 
        return output


