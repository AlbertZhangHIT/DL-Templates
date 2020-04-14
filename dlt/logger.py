import logging

def setup_logger(name, log_file, level=logging.INFO):
	""" setup logger as demanded
	"""
	formatter = logging.Formatter("[%(asctime)s] - %(message)s", "%Y/%m/%d %H:%M:%S")
	handler = logging.FileHandler(log_file)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger
