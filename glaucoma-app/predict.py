import numpy as np

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


def SegmentationResult(filename):
    CATEGORIES = ["Glaucoma", "Non-Glaucoma"]
    pass
    


                 
if __name__ == '__main__':
    from pprint import pprint
    
    filename = 'glaucoma-cases/V0001.jpg'
    results = SegmentationResult(filename)
    pprint(results)
