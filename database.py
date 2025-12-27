import os
import pymongo

def get_database():
    CONNECTION_STRING = os.getenv("MONGO_URI")
    
    client = pymongo.MongoClient(CONNECTION_STRING)
    return client["job_datalake_buffer"]