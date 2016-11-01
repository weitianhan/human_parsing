import os

lis = os.listdir('./annotation')
output = open('train.txt', 'w')
for item in lis:
    output.write(item+'\n')
