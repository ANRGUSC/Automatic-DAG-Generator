import os
import time
import shutil

def task(filename,pathin,pathout):
	execution_time = 1.3
	timeout = time.time() + execution_time
	while time.time() < timeout:
		1+1
	file_name = filename.split('.')[0]
	output1 = ''

	for i in range(30):
		output1 = file_name +'_'+str(i)+'.txt'
		if(not os.path.exists(os.path.join(pathout,output1))):
			break

	input_path = os.path.join(pathin,filename)
	output_path = os.path.join(pathout,output1)

	shutil.copyfile(input_path,output_path)

	return [output_path]

def main():
	pass
