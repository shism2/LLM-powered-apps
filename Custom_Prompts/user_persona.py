from pydantic import BaseModel, Field
import json 
from typing import List, Tuple, Literal
from enum import Enum
from enums import Nations, Gender, Languages, Devices, Characteristics, Hobbies, Education_level
from faker import Faker
import random

# class UserPersona(BaseModel):   # Enum version
#     '''Information about a person.'''
#     gender : Gender = Field(default=Gender.unknown, description='gender of user')
#     # gender : Literal[*list(Gender.__members__.keys())] = Field(default='unknown', description='gender of user')
#     name: str = Field(default='unknown', description='name of user')
#     language : Languages = Field(default=Languages.unknown, description='spoken language of user')
#     # language : Literal[*list(Languages.__members__.keys())] = Field(default='unknown', description='spoken language of user')
#     Nationality: Nations = Field(default=Nations.unknown, description='current location of user')
#     # Nationality: Literal[*list(Nations.__members__.keys())] = Field(default='unknown', description='current location of user')
#     age : int = Field(default=30, gt=5, lt=100, description='age of user')
#     hobbies : List[Hobbies] = Field(default=[Hobbies.unknown], min_items=1, max_items=5, description='hobbies of user')
#     # hobbies : List[Literal[*list(Hobbies.__members__.keys())]] = Field(default=['unknown'], min_items=1, max_items=5, description='hobbies of user')
#     talkative : bool = Field(default=True, description='whether or not user is willing to do small talk which is not related to current topic')
#     characteristics : List[Characteristics] = Field(default=[Characteristics.unknown], min_items=1, max_items=5, description="user's personality")
#     # characteristics : List[Literal[*list(Characteristics.__members__.keys())]] = Field(default=['unknown'], min_items=1, max_items=5, description="user's personality")
#     education_level : Education_level = Field(default=Education_level.unknown, description="user's educational level")
#     # education_level : Literal[*list(Education_level.__members__.keys())] = Field(default='unknown', description="user's educational level")

#     class Config:
#         use_enum_values = False
#         extra = 'forbid'


#     def json(self, **kwargs):  
#         model_dict = super().dict(**kwargs)  
#         for key, value in model_dict.items():  
#             attr = self.__getattribute__(key)  
#             if (not isinstance(attr, str)) and ('__iter__' in dir(attr)):
#                 attr = [x.name if isinstance(x, Enum) else x for x in attr]
#                 model_dict[key] = attr
#             else:
#                 if isinstance(attr, Enum):  
#                     model_dict[key] = attr.name  
#         return json.dumps(model_dict)  


class UserPersona(BaseModel):  # Literal version
    '''Information about a person.'''
    # gender : Gender = Field(default=Gender.unknown, description='gender of user')
    gender : Literal[*list(Gender.__members__.keys())] = Field(default='unknown', description='gender of user')
    name: str = Field(default='unknown', description='name of user')
    # language : Languages = Field(default=Languages.unknown, description='spoken language of user')
    language : Literal[*list(Languages.__members__.keys())] = Field(default='unknown', description='spoken language of user')
    # Nationality: Nations = Field(default=Nations.unknown, description='nationality of user')
    nationality: Literal[*list(Nations.__members__.keys())] = Field(default='unknown', description='nationality of user')
    age : int = Field(default=30, gt=5, lt=100, description='age of user')
    # hobbies : List[Hobbies] = Field(default=[Hobbies.unknown], min_items=1, max_items=5, description='hobbies of user')
    hobbies : List[Literal[*list(Hobbies.__members__.keys())]] = Field(default=['unknown'], min_items=1, max_items=5, description='hobbies of user')
    talkative : bool = Field(default=True, description='whether or not user is willing to do small talk which is not related to current topic')
    # characteristics : List[Characteristics] = Field(default=[Characteristics.unknown], min_items=1, max_items=5, description="user's personality")
    characteristics : List[Literal[*list(Characteristics.__members__.keys())]] = Field(default=['unknown'], min_items=1, max_items=5, description="user's personality")
    # education_level : Education_level = Field(default=Education_level.unknown, description="user's educational level")
    education_level : Literal[*list(Education_level.__members__.keys())] = Field(default='unknown', description="user's educational level")

    class Config:
        use_enum_values = False
        extra = 'forbid'


    def json(self, **kwargs):  
        model_dict = super().dict(**kwargs)  
        for key, value in model_dict.items():  
            attr = self.__getattribute__(key)  
            if (not isinstance(attr, str)) and ('__iter__' in dir(attr)):
                attr = [x.name if isinstance(x, Enum) else x for x in attr]
                model_dict[key] = attr
            else:
                if isinstance(attr, Enum):  
                    model_dict[key] = attr.name  
        return json.dumps(model_dict)  


class UserPersonaDescription(BaseModel):
    '''Descriptive sentence of a persona.'''
    description_of_persona: str = Field(description='descriptive sentence of a persona')

    class Config:
        use_enum_values = False
        extra = 'forbid'


class UserProfile(BaseModel):
    '''User profile consisting of original persona, refined persona by a skillful profiler, and description refined persona'''
    original_persona: UserPersona= Field(description="original persona") 
    refined_persona: UserPersona = Field(description="refined persona") 
    refined_persona_description: UserPersonaDescription = Field(description="description of refined persona") 

    class Config:
        use_enum_values = False
        extra = 'forbid'


# def get_random_user_persona(): # Enum version
#     fake = Faker()
#     def filter_out_list(lst):
#         if 0 in lst:
#             return [0]
#         else:
#             return list(set(lst))

#     inputs = {}
#     inputs['gender'] = random.randint(0, len(Gender.__members__)-1)
#     if inputs['gender'] == 1:
#         inputs['name'] = fake.name_male() 
#     elif inputs['gender'] == 2:
#         inputs['name'] = fake.name_female() 
#     else: 
#         inputs['name'] = fake.name()
#     # inputs['language'] = random.randint(0, len(Languages.__members__)-1)
#     inputs['language'] = 1
#     inputs['Nationality'] = random.randint(0, len(Nations.__members__)-1)
#     inputs['age'] = random.randint(5, 99)
#     inputs['hobbies'] = filter_out_list([random.randint(0, len(Hobbies.__members__)-1) for _ in range(0, random.randint(1, UserPersona.__fields__['hobbies'].field_info.max_items))])
#     inputs['talkative'] = fake.boolean()
#     inputs['characteristics'] = filter_out_list([random.randint(0, len(Characteristics.__members__)-1) for _ in range(0, random.randint(1, UserPersona.__fields__['characteristics'].field_info.max_items))])
#     inputs['education_level'] = random.randint(0, len(Education_level.__members__)-1)
#     return UserPersona(**inputs)


def get_random_user_persona():  # Literal version
    fake = Faker()
    def filter_out_list(lst):
        if 0 in lst:
            return [0]
        else:
            return list(set(lst))

    inputs = {}
    inputs['gender'] =  list(Gender.__members__.keys())[random.randint(0, len(Gender.__members__)-1)]
    if inputs['gender'] == 'mail':
        inputs['name'] = fake.name_male() 
    elif inputs['gender'] == 'female':
        inputs['name'] = fake.name_female() 
    else: 
        inputs['name'] = fake.name()
    # inputs['language'] = random.randint(0, len(Languages.__members__)-1)
    inputs['language'] = 'English'
    inputs['nationality'] = list(Nations.__members__.keys())[random.randint(0, len(Nations.__members__)-1)]
    inputs['age'] = random.randint(5, 99)
    inputs['hobbies'] = filter_out_list([  list(Hobbies.__members__.keys())[random.randint(0, len(Hobbies.__members__)-1)]  for _ in range(0, random.randint(1, UserPersona.__fields__['hobbies'].field_info.max_items))])
    inputs['talkative'] = fake.boolean()
    inputs['characteristics'] = filter_out_list([  list(Characteristics.__members__.keys())[random.randint(0, len(Characteristics.__members__)-1)]  for _ in range(0, random.randint(1, UserPersona.__fields__['characteristics'].field_info.max_items))])
    inputs['education_level'] = list(Education_level.__members__.keys())[random.randint(0, len(Education_level.__members__)-1)] 
    return UserPersona(**inputs)




if __name__ == '__main__':
    print("This is user_persona.py executed directly")