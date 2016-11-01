import os
import sys

lis = os.listdir('./annotation')
rate = len(lis) / 203 
i = 0
while (i < len(lis)):
    os.rename('./annotation/%s' % lis[i], './test/%s' % lis[i])
    i += rate
