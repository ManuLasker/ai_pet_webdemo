import logging
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=BASIC_FORMAT, level=logging.INFO)

logger = logging.getLogger()