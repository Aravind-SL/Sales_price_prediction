import sys
import os
import pandas as pd

from src.utils import load_model
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def predict(self, feature):
        try:
            model_path= os.path.join('artifacts', 'model.pkl')
            preprocessor_path= os.path.join('artifacts', 'preprocessor.pkl')
            model_obj= load_model(model_path)
            preprocessor_obj= load_model(preprocessor_path)

            logging.info('Loaded objet successfully.')

            scaled_data= preprocessor_obj.transform(feature)
            predicted_value= model_obj.predict(scaled_data)

            return predicted_value

        except Exception as e:
            raise CustomException(e, sys)

