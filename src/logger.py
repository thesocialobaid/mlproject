import logging 
import os 

from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE) #cwd means current working directory, whatever logs are there they get into the log path 
os.makedirs(logs_path,exist_ok=True) # if the logs folder is not there it will create it

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE) 

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)

if __name__ == "__main__":
    logging.info("Logging has started")
    