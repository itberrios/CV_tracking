'''
This script contains various utility functions for working with the KITTI dataset
'''

def test_func(a, b):
    ''' Trivial function to test imports in Google Colab '''
    return a + b

# ============================================================================================
# GPS/IMU functions

def get_oxts(oxt_path):
    ''' Obtains the oxt info from a single oxt path '''
    with open(oxts_data_paths[0]) as f:
        oxts = f.readlines()
        
    oxts = oxts[0].strip().split(' ')
    oxts = np.array(oxts).astype(float)

    return oxts
