"""
This is the code for directory parsing
in order to make label files for 
UCF-CRIME datastet
"""

import glob
import os
a = [name for name in os.listdir("/media/ihsan/HDDP12/Workspace/ClassIE803/HAR/UCFCrime/MAR_DATA/Normal") if name.endswith(".mp4")]
a.sort()
print(a)
with open('1-listfile.txt', 'w') as filehandle:
    for listitem in a:
        filehandle.write('Normal/%s\n' % listitem)
print('done')
