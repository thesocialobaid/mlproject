# A simple web application which will be interacting with the input data 
# Form for input data for predicting the student data, 
# Captures the data and works with the preprocesser in the pkl file. 

import sys 
import pandas as pd 
from src.exception import CustomException
from src.utils import load_object 


class PredictPipeline:
    def __init__(self): 
        pass 
    
    def predict(self,features): # just like model prediction and doing the predictions 
        model_path = 'artifacts/model.pkl'
        preprocessor_path = 'artifacts/preprocessor.pkl'
        model=load_object(file_path=model_path)    # calling the load object function to load the model and preprocessor
        preprocessor=load_object(file_path=preprocessor_path)
        data_scaled = preprocessor.transform(features) # transforming the features using the preprocessor
        preds = model.predict(data_scaled) # making the predictions using the model
        return preds
    
        


class CustomData:        # responsibile in mapping the inputs in the html to the one for the model 
    def __init__(   self, 
            gender: str,
            race_ethnicity:int,
            parental_level_of_education: str, 
            lunch:str,
            test_preparation_course:str,
            reading_score:int, 
            writing_score:int): 
        
        self.gender = gender 
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
    
    def get_data_as_dataframe(self):
        try: 
            custom_data_input_dict = {
                "gender": [self.gender], 
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]   
            }
            return pd.DataFrame(custom_data_input_dict) # this will return the dataframe of the input data
        
        except Exception as e: 
            raise CustomException(e, sys) 
        