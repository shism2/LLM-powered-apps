from enum import Enum


def get_enum_schemas():
    def get_enum_schema(enum_classes):  
        schemas = []  
        for enum_class in enum_classes:  
            schema = f"Enum class '{enum_class.__name__}': " + ", ".join([f"{member.name} = {member.value}" for member in enum_class])  
            schemas.append(schema)  
        return "\n\n".join(schemas) 

    return get_enum_schema([Nations, Languages, Gender, Devices, Characteristics, Hobbies, Education_level, Specialties])   


class Nations(Enum):
    unknown = 0
    United_States_of_America = 1
    China = 2
    Japan = 3
    Germany = 4 
    India = 5
    United_Kingdom = 6
    France = 7
    Brazil = 8
    Italy = 9
    Canada = 10
    South_Korea = 11
    Russia = 12
    Australia = 13 
    Spain = 14
    Mexico = 15
    Indonesia = 16 
    Netherlands = 17
    Saudi_Arabia = 18
    Turkey = 19
    Switzerland = 20
    Taiwan = 21
    Sweden = 22
    Poland = 23
    Belgium = 24
    Argentina = 25
    Thailand = 26
    Iran = 27
    Austria = 28 
    Norway = 29
    United_Arab_Emirates = 30


class Languages(Enum):
    unknown = 0
    English = 1
    Mandarin = 2
    Japanese = 3
    Hindi = 4
    French = 5
    Portuguese = 6
    Italian = 7
    Korean = 8
    Russian = 9
    Indonesian = 10
    Dutch = 11
    Arabic = 12
    Turkish = 13
    German = 14
    Romansh = 15
    Swedish = 16
    Polish = 17
    Spanish = 18
    Thai = 19
    Persian = 20 
    Norwegian = 21


class Gender(Enum):
    unknown = 0
    male = 1
    female = 2
    Non_binary = 3


class Devices(Enum):
    unknown = 0
    air_conditioner = 1
    washing_machine = 2
    dryer = 3
    styler = 4
    water_purifier = 5
    air_purifier = 6
    cooktop = 7
    standby_me = 8



class Characteristics(Enum):
    unknown = 0
    Diligent = 1
    Enthusiastic = 2
    Humble = 3
    Intuitive = 4
    Jovial = 5
    Kind_hearted = 6 
    Loyal = 7
    Meticulous = 8
    Nurturing = 9
    Optimistic = 10
    Patient = 11
    Resourceful = 12
    Sincere = 13
    Timid = 14
    Understanding = 15
    Vibrant = 16
    Witty = 17
    Xenial = 18
    Yielding = 19
    Zealous = 20
    Arrogant = 21
    Boisterous = 22
    Cynical = 23
    Dismissive = 24 
    Evasive = 25
    Fickle = 26
    Grumpy = 27
    Impulsive =28     
    Compassionate = 29
    NotDiligent = 30
    NotEnthusiastic = 31
    NotHumble = 32
    NotIntuitive = 33
    NotJovial = 34
    NotKind_hearted = 35 
    NotLoyal = 36
    NotMeticulous = 37
    NotNurturing = 38
    NotOptimistic = 39
    NotPatient = 40
    NotResourceful = 41
    NotSincere = 42
    NotTimid = 43
    NotUnderstanding = 44
    NotVibrant = 45
    NotWitty = 46
    NotXenial = 47
    NotYielding = 48
    NotZealous = 49
    NotArrogant = 50
    NotBoisterous = 51
    NotCynical = 52
    NotDismissive = 53 
    NotEvasive = 54
    NotFickle = 55
    NotGrumpy = 56
    NotImpulsive =57     
    NotCompassionate = 58

class Hobbies(Enum):
    unknown = 0
    playing_with_electronic_devices = 1
    sports = 2
    watching_movies = 3
    reading = 4
    gardening = 5
    painting = 6
    hiking = 7
    knitting = 8
    cooking = 9
    photography = 10
    playing_guitar = 11
    traveling = 12
    swimming = 13
    dancing = 14
    blogging = 15
    birdwatching = 16
    writing_poetry = 17
    fishing = 18
    cycling = 19
    playing_chess = 20
    baking = 21
    running = 22
    collecting_stamps = 23
    yoga = 24
    camping = 25
    DIY_projects = 26
    singing = 27
    pottery = 28
    skateboarding = 29
    astronomy = 30
    playing_video_games = 31


class Education_level(Enum):
    unknown = 0
    kindergarten_level = 1
    elementary_school_level = 2
    middle_school_level = 3
    high_school_level = 4
    undergraduate_college_level = 5
    master_degree_level = 6
    phd_degree_level = 7
    genius_level = 8


class Specialties(Enum):
    general_area = 0
    electronic_devices = 1
    air_conditioner = 2
    washing_machine = 3
    household_jobs = 4
    pets = 5
    fruits = 6
    vegetables = 7
    authors = 8
    cities = 9
    musicals = 10
    movies = 11
    historical_figures = 12
    sports = 13
    cars = 14
    flowers = 15
    animals = 16
    artists = 17
    books = 18
    countries = 19
    languages = 20
    foods = 21
    beverages = 22
    scientists = 23
    instruments = 24
    painters = 25
    philosophers = 26
    universities = 27
    mountains = 28
    rivers = 29
    oceans = 30
    deserts = 31
    islands = 32
    video_games = 33
    software = 34
    websites = 35
    novels = 36
    musicians = 37
    board_games = 38
    diseases = 39
    trees = 40
    birds = 41
    insects = 42
    constellations = 43
    composers = 44
    poets = 45
    architects = 46
    ancient_civilizations = 47
    sports_teams = 48
    music_genres = 49
    dances = 50
    fashion_designers = 51
    paintings = 52
    sculptures = 53
    mythological_figures = 54
    festivals = 55




if __name__ == '__main__':
    print("This is enums.py executed directly")