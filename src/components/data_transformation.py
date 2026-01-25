# Feature Engineering and Data Transformation Module 

import sys 
import os 
from dataclasses import dataclass 

import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer    # Column Transformer helps to apply different transformations to different columns.
from sklearn.impute import SimpleImputer        #Simple Imputer helps to handle missing values.
from sklearn.pipeline import Pipeline           # importing the pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder # One Hot Encoding is best for "Nominal Data "- categories with no natural order
from sklearn.preprocessing import StandardScaler # Standard Scaler is used to standardize the features by removing the mean and scaling to unit variance.

from src.exception import CustomException 
from src.logger import logging 
from src.util import save_object

@dataclass # data class is used to create classes that primarily store data with less boilerplate code.
class DataTransformationConfig: #class for data-transformation config 
    preprocessor_obj_file_path= os.path.join('artifacts','preprocessor.pkl')  # a pkl file is a serialized Python object created using the pickle module. In Python "pickling" is the process of converting a Python object hierarchy into a byte stream 
    
class DataTransfrormation: 
    def __init__(self): 
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transfomer_object(self): # files responsible for converting categorical features into numerical features 
        '''
        Docstring for get_data_transfomer_object
        This function is responsible for data transformation 
        :param self: Description
        '''
        try: 
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                "gender", 
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps=[
                 ("imputer", SimpleImputer(strategy="median")), # a imputer trys to add values to the missing columns based on the information based on the previous data in the columns. A median strategy means that it sorts the data, identifies the median and replaces every empty space with that median. 
                 ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps = [ 
                    ("imputer", SimpleImputer(strategy="most_frequent")), # for categorical data we use most frequent strategy to fill the missing values. 
                    ("encoder", OneHotEncoder()), # one hot encoder is used to convert categorical data into numerical data by creating multiple binary columns for each category. 
                ]
            )
            logging.info("Numrical columns scaling completed")
            logging.info("Categorical columns encoding completed")
            
            preprocessor = ColumnTransformer(
                [ ("num_pipeline",num_pipeline, numerical_columns), 
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                    ]
            )
            
            return preprocessor
        except Exception as e:
         raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path, test_path): 
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            logging.info("Obtaining preprocessing object")
            
            preprocessor_obj = self.get_data_transfomer_object()
            
            target_column_name = "math_score"  #the purpose is to predict the math score based on other scores and categorical data.
            
            
            # Divide the train dataset into independent and dependent features
            input_feature_train_df = train_df.drop([target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide the test dataset into independent and dependent features
            input_feature_test_df = test_df.drop([target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")

            # Fit and Transform on training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            
            # ONLY Transform on testing data to prevent data leakage
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            # Concatenating the features with the target column
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object.")
            
            # Save the pickle file
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

       
            
            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e: 
            raise CustomException(e,sys)
        
        