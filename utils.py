import pdb
a=[1, 2, 3, 4, 5]
try:
    from . import config
except ImportError:
    pdb.set_trace()