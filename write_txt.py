# Program to show various ways to read and 
# write data in a file.
import os,glob
datafile = glob.glob('../tfrecord_x1_main/tfrecord_x1_0/*')
datafile1 = glob.glob('../tfrecord_x1_main/tfrecord_x1_1/*')
file1 = open("myfile.txt","w") 
L=[]
for i in range(len(datafile)):
    L.append(datafile[i].partition("/")[-1].partition("/")[-1]+" 0\n")
for i in range(len(datafile1)):
    L.append(datafile1[i].partition("/")[-1].partition("/")[-1]+" 1\n")
# \n is placed to indicate EOL (End of Line) 
file1.writelines(L) 
file1.close() #to change file access modes 
