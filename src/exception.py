import sys
import logging

def error_message_detail(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    lineno = exc_tb.tb_lineno
    error_message = "Error occured in python script \nname : {0}\nLine number: {1}\nError message: {2}".format(
        filename, lineno, error
    )

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Division by zero")
        raise CustomException(e, sys)
