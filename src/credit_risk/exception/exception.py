"""Legacy pipeline exception with source-location context."""

import sys


class customexception(Exception):
    def __init__(self, error_message, error_details: sys):
        self.error_message = error_message
        _, _, exc_tb = error_details.exc_info()
        print(exc_tb)

        self.lineno = exc_tb.tb_lineno
        self.file_name = exc_tb.tb_frame.f_code.co_filename

    def __str__(self):
        return f"Error occured in python script name [{self.file_name}] line number [{self.lineno}] error message [{str(self.error_message)}]"
