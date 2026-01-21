import sys  # the sys library provides various functions and variables that are used to manipulate different parts of the Python runtime environment. 
import logging 

# function to generate error message details
def error_message_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info() # exc_info() function returns a tuple containing information about the most recent exception caught by an except clause in the current thread.
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

#class for custom exception handling
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) #we need to initialize, inherit the init function.
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        logging.info("Divide by zero exception occurred")
        raise CustomException(e, sys)
    
      