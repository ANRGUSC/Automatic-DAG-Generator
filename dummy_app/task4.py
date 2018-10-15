import os
import time
import shutil
import uuid

def task(filename,pathin,pathout):
	execution_time = 1.1
	timeout = time.time() + execution_time
	while time.time() < timeout:
		1+1
	file_name = filename[0].split('.')[0]
	output1 = file_name +'_'+str(uuid.uuid1())+'.txt'

	input_path = os.path.join(pathin,filename[0])
	output_path = os.path.join(pathout,output1)

	shutil.copyfile(input_path,output_path)

	return [output_path]

def main():
	filelist = ['1botnet.ipsum']
	outpath = os.path.join(os.path.dirname(__file__), 'sample_input/')
	outfile = task(filelist, outpath, outpath)
	return outfile
