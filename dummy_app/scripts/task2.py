import os
import time
import shutil

def task(filename,pathin,pathout):
	execution_time = 2.0
	execution_time
	timeout = time.time() + execution_time
	while time.time() < timeout:
		1+1
	task_name = os.path.basename(__file__).split('.')[0]
	print('-------------------------')
	print(task_name)
	print(filename)
	print(pathin)
	print(pathout)
	print('-------------------------')
	if isinstance(filename, list):
		output1 = [file.split('.')[0] +'_'+task_name+'.txt' for file in filename]
		input_file = filename[0].split('_')[0]
	elif not isinstance(filename, list):
		output1=[filename.split('.')[0] +'_'+task_name+'.txt']
		input_file = filename.split('_')[0]
	print(output1)
	print(input_file)
	output_fname=[f.split('.')[0].split('_')[-1] for f in output1]
	output_name='_'.join(output_fname)
	output_name=input_file+'_'+output_name
	print(output_name)
	f = open('centralized_scheduler/communication.txt', 'r')
	total_info = f.read().splitlines()
	f.close()
	comm = dict()
	for line in total_info:
		src = line.strip().split(' ')[0]
		dest_info = line.split(' ')[1:-1]
		if len(dest_info)>0:
			comm[src] = dest_info
	print('-------------------------##')
	print(comm)
	print(comm.keys())
	if task_name in comm.keys():
		print(comm[task_name])
		dest=[x.split('_')[0] for x in comm[task_name]]
		print(dest)
		comm_data=[float(x.split('_')[1]) for x in comm[task_name]]
		print(comm_data)
	file_size = '10'
	if not os.path.isdir(pathout):
		os.makedirs(pathout, exist_ok=True)
	output_path=os.path.join(pathout,output_name) 
	print(output_path)
	bash_script='centralized_scheduler/generate_random_files.sh'+' '+output_path+' '+file_size
	print(bash_script)
	os.system(bash_script)

	return output_path

def main():
	filelist = '1botnet.ipsum'
	outpath = os.path.join(os.path.dirname(__file__), 'sample_input/')
	outfile = task(filelist, outpath, outpath)
	return outfile
