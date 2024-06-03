import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql
import pickle

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=user,
            db=db
        )
        logging.info(f"Connection Established: {mydb}")
        df=pd.read_sql_query('select * from data', mydb)
        print(df.head())

        return df
        
    except Exception as e:
        logging.error(f"Error establishing connection: {e}")
        raise


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)