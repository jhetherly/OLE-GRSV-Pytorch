import sys
import logging
import logging.handlers
import inspect


def setup_logger(name, log_file=None, level=logging.INFO):
	"""Either retrieves or creates a new logger"""

	logger = logging.getLogger(name)
	logger.setLevel(level)
	# check if logger doesn't already exist
	if len(logger.handlers) > 0:
		return logger
	
	FORMAT = "%(levelname)s | %(funcName)s | %(asctime)s | %(message)s"
	DATE_FORMAT = "%y/%m/%d %H:%M:%S %Z"
	console_formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)
	
	FORMAT = "%(levelname)s | %(filename)s:%(lineno)d | %(funcName)s | %(process)d:%(thread)d | %(asctime)s | %(message)s"
	file_formatter = logging.Formatter(FORMAT, datefmt=DATE_FORMAT)

	if log_file is None:
		frame = inspect.stack()[1]
		module = inspect.getmodule(frame[0])
		filename = module.__file__
		log_file = filename + ".log"
	file_handler = logging.handlers.RotatingFileHandler(log_file, mode='w', maxBytes=10000000, backupCount=5)        
	file_handler.setFormatter(file_formatter)

	console_handler = logging.StreamHandler(sys.stderr)
	console_handler.setFormatter(console_formatter)
	
	logger.addHandler(file_handler)
	logger.addHandler(console_handler)
	
	return logger
