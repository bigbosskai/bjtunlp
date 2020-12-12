from .nlp import BJTUNLP
from warnings import simplefilter
import warnings
warnings.filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)
__all__ = ['BJTUNLP']