import os
import logging
from ..utils.management.decorators import (
    log
)

#################
#   Variables   #
#################

logger = logging.getLogger(__name__)

#################
#   Functions   #  
#################

@log(include_timer=True)
def train_model():
    pass

def plot_training_history():
    pass

@log(include_timer=True)
def setup_training_pipeline():
    pass

