from transformers.utils import logging

logging.set_verbosity_info()

# Imprimir la ubicación de la caché
from transformers import HfFolder
print(HfFolder.get_cache_home())
