"""
This code uses the output DAG of ``rand_task_gen.py``(by Diyi), which is used to generate a random DAG, to generate the corresponding dummy application working for Jupiter (version 3) , 

"""
__author__ = "Quynh Nguyen, Jiatong Wang, Diyi Hu and Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2019, Autonomous Networks Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"


import argparse
import numpy as np
import random
from functools import reduce
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pylab as plt
import yaml
import os
import json
import shutil
from collections import defaultdict
import random

EPSILON = 1e-2

def parse_args(conf_file):
	parser = argparse.ArgumentParser(description='generate random task graphs')
	#parser.add_argument("--conf",required=True,type=str,help='yaml file specifying task_dag generation parameters')
	parser.add_argument('--conf', nargs='?', const=conf_file, type=str,help='yaml file specifying task_dag generation parameters')
	return parser.parse_args()

def random_list(depth,total_num,width_min,width_max):
    list_t = []
    if total_num>= (depth-2)*width_min+2 and total_num<=(depth-2)*width_max+2:
        list_t.append(1)
        for i in range(depth-2):
            list_t.append(2)
        list_t.append(1)
        #print(list_t)
        for i in range(total_num-sum(list_t)+2):
            while True:
                tmp = random.randint(1,len(list_t)-2)
                if list_t[tmp]+1 > 4:
                    pass
                else:
                    list_t[tmp] = list_t[tmp] + 1
                    break
    else:
        list_t.append(1)
        num_tries = 30
        for num in range(0,num_tries):
            t = random.randint(width_min,width_max)
            # print('-------')
            # print(t)
            # print(total_num)
            # print(list_t)
            a = sum(list_t)-1+t
            # print(a)
            # print(width_min)
            b = total_num -(sum(list_t)-1)
            # print(b)
            # print(width_max)

            if (sum(list_t)-1+t)<total_num:
                list_t.append(t)
            elif  total_num -(sum(list_t)-1) >=width_min and total_num -(sum(list_t)-1)<=width_max :
                list_t.append(total_num -(sum(list_t)-1))
                break
            else:
                print('something wrong')
                pass
        list_t.append(1)
    return list_t

def gen_task_nodes(depth,total_num,width_min,width_max):
	#num_levels = depth+2		# 2: 1 for entry task, 1 for exit task
	num_list = random_list(depth+2,total_num,width_min,width_max)
	print(num_list)
	num_levels = len(num_list)
	num_nodes_per_level = np.array(num_list)
	#num_nodes_per_level = np.array([random.randint(width_min,width_max) for i in range(num_levels)])
	num_nodes_per_level[0] = 1.
	num_nodes_per_level[-1] = 1.
	num_nodes = num_nodes_per_level.sum()
	level_per_task = reduce(lambda a,b:a+b, [[enum]*val for enum,val in enumerate(num_nodes_per_level)],[])
	#e.g. [0,1,2,2,3,3,3,3,4]
	level_per_task = {i:level_per_task[i] for i in range(num_nodes)}
	#level_per_task in the format of {task_i: level_of_i}
	task_per_level = {i:[] for i in range(num_levels)}
	for ti,li in level_per_task.items():
		task_per_level[li] += [ti]
		# task_per_level in the format of {level_i:[tasks_in_level_i]}
	return task_per_level, level_per_task

def gen_task_links(deg_mu,deg_sigma,task_per_level,level_per_task,delta_lvl=2):
	num_tasks = len(level_per_task)
	num_level = len(task_per_level)
	neighs_top_down = {t:np.array([]) for t in range(num_tasks)}
	neighs_down_top = {t:np.array([]) for t in range(num_tasks)}
	deg = np.random.normal(deg_mu,deg_sigma,num_tasks)
	deg2 = (deg/2.).astype(np.int)
	deg2 = np.clip(deg2,1,20)
	#add edges from top to down with deg2, then bottom-up with deg2
	edges = []
	# ---- top-down ----
	for ti in range(num_tasks):
		if level_per_task[ti] == num_level-1:	# exit task is a sink
			continue
		ti_lvl = level_per_task[ti]
		child_pool = []
		for li,tli in task_per_level.items():
			if li <= ti_lvl or li > ti_lvl+delta_lvl:
				continue
			child_pool += tli
		neighs_top_down[ti] = np.random.choice(child_pool,min(deg2[ti],len(child_pool)),replace=False)
		edges += [(str(ti),str(ci)) for ci in neighs_top_down[ti]]
	# ---- down-top ----
	for ti in reversed(range(num_tasks)):
		if level_per_task[ti] == 0:
			continue
		ti_lvl = level_per_task[ti]
		child_pool = []
		for li,tli in task_per_level.items():
			if li >= ti_lvl or li < ti_lvl-delta_lvl:
				continue
			child_pool += tli
		neighs_down_top[ti] = np.random.choice(child_pool,min(deg2[ti],len(child_pool)),replace=False)
		edges += [(str(ci),str(ti)) for ci in neighs_down_top[ti]]
	return list(set(edges)),neighs_top_down,neighs_down_top

def gen_attr(tasks,edges,ccr,comp_mu,comp_sigma,link_comm_sigma):
	task_comp = np.clip(np.random.normal(comp_mu,comp_sigma,len(tasks)), EPSILON, comp_mu+10*comp_sigma)
	link_comm = np.zeros(len(edges))
	link_comm_mu = comp_mu * ccr
	#link_comm is the data transmitted on links, comp is the computation workload. They both follow normal distribution. ccr is a constant
	link_comm = np.clip(np.random.normal(link_comm_mu,link_comm_sigma*link_comm_mu,len(edges)),EPSILON, link_comm_mu+10*link_comm_sigma*link_comm_mu)
	return task_comp,link_comm

def plot_dag(dag,dag_path_plot):
	pos = graphviz_layout(dag,prog='dot')
	node_labels = {n:'{}-{:3.1f}'.format(n,d['comp']) for n,d in dag.nodes(data=True)}
	edge_labels = {(e1,e2):'{:4.2f}'.format(d['data']) for e1,e2,d in dag.edges(data=True)}
	plt.clf()
	nx.draw(dag,pos=pos,labels=node_labels,font_size=8)
	nx.draw_networkx_edge_labels(dag,pos,edge_labels=edge_labels,label_pos=0.75,font_size=6)
	plt.savefig(dag_path_plot)



def prepare_task_dag(config_yml,dag_path_plot):
	with open(config_yml) as f_config:
		config = yaml.load(f_config)
	#--- generate task graph ---

	task_per_level,level_per_task = gen_task_nodes(config['depth'],config['total_num'],config['width_min'],config['width_max'])
	edges,adj_list_top_down,adj_list_down_top = gen_task_links(config['deg_mu'],config['deg_sigma'],task_per_level,level_per_task)
	task_comp,link_comm = gen_attr(np.arange(len(level_per_task)),edges,config['ccr'],config['comp_mu'],config['comp_sigma'],config['link_comm_sigma'])
	edge_d = [(e[0],e[1],{'data':link_comm[i]}) for i,e in enumerate(edges)]
	dag = nx.DiGraph()
	dag.add_edges_from(edge_d)
	for i,t in enumerate(task_comp):
		dag.node[str(i)]['comp'] = t
		##NOTE: actually it should not be called 'comp', but 'computation data amount'!!!
	if dag_path_plot is not None:
		plot_dag(dag,dag_path_plot)
	#print(dag.graph)
	
	return dag

#Generate configuration.txt
def generate_config(dag,app_path):
	f = open(app_path, 'w')
	total_node = len(dag.nodes())
	f.write(str(total_node) + "\n")
	task_dict = {}
	for j in dag.nodes():
		task_dict[j] = 0
	task_dict['0'] = 1
	for e0, e1 in dag.edges():
		if e1 in task_dict.keys():
			task_dict[e1] += 1

	data = dict()
	for i in dag.nodes():
		if i not in data.keys():
			data[i] = ""
		data[i] += str(task_dict[i]) + " "
		#data[i] += "true " ## send single output to all the children tasks
		data[i] += "false " ## send all children output to all the children tasks correspondingly
		for e0, e1 in dag.edges():
			if i == e0:
				data[i] +="task" + e1 + " "
		if int(i) == total_node - 1:
			data[i] += "home"
	for i in range(len(data)):
		f.write("task" + str(i) + " ")
		f.write(data[str(i)])
		f.write('\n')

	f.close()

def generate_communication(dag,app_path):
	f = open(app_path,'w')
	for i in dag.nodes():
		f.write("task"+i+ " ")
		for e0,e1,d in dag.edges(data=True):
			if i == e0:
				f.write('task'+e1+'-'+str(round(d['data'],2))+' ') #KB
		f.write("\n")
	f.close()


def generate_scripts(dag,config_path,script_path,app_path,sample_path):
    

    print('------ Read input parameters from DAG  -------------')
    sorted_list = sorted(dag.nodes(data=True), key=lambda x: x[0], reverse=False)
    f = open(config_path, 'r')
    total_line = f.readlines()
    del total_line[0]
    f.close()


    for i in range(len(total_line)):
        tmp = total_line[i].split(' ')
        filename = tmp[0] + ".c"
        num = int(tmp[1])
        file_path = script_path + filename

        f = open(file_path, 'w')
        f.write("#include<stdio.h>\n")
        f.write("#include<stdlib.h>\n")
        f.write("#include<string.h>\n")
        f.write("#include<math.h>\n")
        f.write("#include<time.h>\n")
        f.write("#include<sys/stat.h>\n")
        f.write("#include<sys/types.h>\n")
        f.write("#define PATH_MAX 128\\n")
        f.write("#include<libgen.h>\n")

        f.write("int tot=i;\n")
        f.write("char filename[128][128];\n")
        f.write("char file_path[128][128];\n")
        f.write("int execution_time;\n")
        f.write("char* task_name;\n")
        f.write("char output1_list[128];\n")
        f.write("char input_file[128];\n")
        f.write("FILE* f;\n")
        f.write("char comm [128][128];\n")
        f.write("char *src[128][20];\n")
        f.write("char* new_path[128];\n")
        f.write("int loop_var;\n")
        f.write("char* ptr;\n")

        f.write("\n")
        f.write("void task(filename,pathin,pathout)\n")
        f.write("{ \n")
        f.write("\texecution_time = rand() % 10;\n")
        f.write('\tprintf("%d",execution_time);\n')
        f.write("\ttime_t timeout;\n")
        f.write("\ttimeout = time(&timeout) + execution_time;\n")

        f.write("\twhile (time.time() < timeout)\n")
        f.write("{ \n")
        f.write("\t\t1+1 ;\n")
        f.write("} \n")

        f.write("\tchar *path = '__FILE__';\n")
        f.write("\tchar *path_cpy = strdup(path);\n")
        f.write("\ttask_name=basename(path_cpy);\n")
        f.write('\tprintf("-------------------------");\n')
        f.write('\tprintf("%s",task_name);\n')
        f.write('\tprintf("%s",filename);\n')
        #f.write("\tprintf(pathin)\n")
        #f.write("\tprint(pathout)\n")
        f.write('\tprintf("-------------------------" );\n')
        f.write("\t\tstrcpy(output1_list,&filename);\n")
        f.write('\t\tstrcat(output1_list,"."");\n')
        f.write("\t\tstrcat(output1_list,&task_name[0]);\n")
        f.write('\t\tstrcat(output1_list,".txt") ;\n')
        f.write("\t\tstrcpy(input_file, &filename);\n")
        f.write('\tprintf("%s",output1_list);\n')
        f.write('\tprintf("%s",input_file);\n')

        f.write('\t\tchar output_name[128]=strcat(input_file,"_");\n')
        f.write("\t\tstrcat(output_name,output_name);")
        f.write('\tprint("%s",output_name);\n')
        f.write('\tprintf("---------------------------");\n')

        f.write("\tchar actualpath [PATH_MAX+1];\n")
        f.write('\tchar *file_comm [PATH_MAX+1]= "communication.txt";\n')
        f.write("\tchar *ptr;;\n")
        f.write("\tptr=realpath(file_comm,actualpath);\n")
        f.write('\tprintf("%s",ptr);\n')

        f.write("\tFILE* f;\n")
        f.write('\tfopen(/centralized_scheduler/communication.txt, "r" );\n')
        f.write("\tfclose(f);\n")

        f.write("\tchar temp_comm [128][128];\n")
        f.write("\tchar comm [128][128];\n")
        f.write('\tmemset(temp_comm, "\0", sizeof(temp_comm));\n')
        f.write("\tmemset(comm, '\0', sizeof(comm));;\n")
        f.write("\tchar line_read [128];\n")
        f.write("\tmemset(line_read, '\0', sizeof(line_read));\n")
        f.write("\tint k = 0;\n")
        f.write("\twhile( fgets( line_read, sizeof(line_read), f ) != NULL )\n")
        f.write("{ \n")
        f.write("\tstrcpy(temp_comm[k], line_read);\n")
        f.write("\tk++;\n")
        f.write("} \n")
        f.write("\tint sum_k = k;\n")
        f.write("\tchar *src[128][20];\n")
        f.write("\tchar *dest_temp[128][20];\n")
        f.write("\tfor(int loop_var=0; loop_var < sum_k; loop_var++)\n")
        f.write("{ \n")
        f.write("\tstrncpy(src[loop_var], temp_comm[loop_var], 6);\n")
        f.write("\tchar *tmp = strchr(temp_comm[loop_var], ' ');\n")
        f.write("\tif(tmp != NULL)\n")
        f.write("\tchar *dest_temp[loop_var]=tmp+1;\n")
        f.write("\tif(strlen(dest_temp[loop_var])>0)\n")
        f.write("\tcomm[loop_var] = dest_temp[loop_var];\n")
        #f.write("} \n")
        f.write('\tprintf("---------------------------");\n')
        f.write('\tprintf("%s",comm[loop_var]);\n')
        f.write('\tprintf("%s",task_name);\n')


        f.write("\tstruct stat pathout_check;\n")
        f.write("\t\tstat(pathout, &pathout_check);\n")
        f.write("\t\tif (!(S_ISDIR(pathout_check.st_mode)));\n")
        f.write("\t\tmkdir(pathout);\n")

        f.write("\tchar output_path=[];\n")
        f.write("\tif strcmp(task_name,src[loop_var])\n")
        f.write("{ \n")
        f.write('\tprintf("%s",comm[loop_var]);\n')
        f.write("\tchar output_list[128][128];\n")
        f.write("\tchar file_size[128][128];\n")
        f.write("\t\tchar new_file[128];\n")
        f.write('\tprintf("The neighor is:");\n')
        f.write('\tprintf("%s",comm[loop_var]);\n')
        f.write('\tprintf("The IDX  is:");\n')
        f.write('\tprintf("%s",src[loop_var]);\n')
        f.write("\tstrcpy(new_file,output_name);\n")
        f.write('\tstrcat(new_file,"_");\n')
        f.write("\tstrcat(new_file,src[loop_var]);\n")
        f.write("\toutput_list[j]=new_file;\n")
        f.write("\tfile_size[j]=comm[j];\n")
        f.write("\tchar* new_path[128]; \n")
        f.write("\tmemset(new_path, '\0', sizeof(new_path));\n")
        f.write("\tstrcpy(new_path,pathout);\n")
        f.write("\tstrcat(new_path,new_file);\n")
        f.write("\tstrcpy(output_path,new_path);\n")
        f.write('\tprintf("%s",new_path);\n')
        f.write("\tchar bash_script[128]; \n")                               
        f.write("\tsystem(bash_script); \n")
        f.write("} \n")                                                     
        

        f.write("\telse if !strcmp(task_name,src[loop_var]) \n")
        f.write("\t \t {  \n") 
        f.write("\tstrcpy(new_file,output_name); \n")
        f.write('\tstrcat(new_file,"_"); \n')
        f.write("\tstrcat(new_file,task_name); \n")
        f.write("\tstrcpy(output_path,new_path);\n")
        f.write("\t file_size=itoa(rand() + 1);\n")
        f.write("\tstrcpy(new_file,output_name); \n")
        f.write("\tsystem(bash_script); \n")                                      
        f.write("\t \t } \n") 
        f.write("\treturn output_path; \n")
        f.write("\t \t } \n") 

        f.write("\n")
        f.write("def main():\n")
        f.write("\tfilelist = '1botnet.ipsum'\n")
        f.write("\toutpath = os.path.join(os.path.dirname(__file__), 'sample_input/')\n")
        f.write("\toutfile = task(filelist, outpath, outpath)\n")
        f.write("\treturn outfile\n")
        f.close()
	shutil.copy('jupiterapp/app_config.ini',app_path)
	shutil.copy('jupiterapp/input_node.txt',app_path) #for WAVE
	shutil.copy('jupiterapp/generate_random_files.sh',script_path)
	os.mkdir(sample_path)
	shutil.copy('jupiterapp/1botnet.ipsum',sample_path)
	shutil.copy('jupiterapp/2botnet.ipsum',sample_path)


def generate_nameconvert(dag,app_path):

	f = open(app_path,'w')
	for i in dag.nodes():
		s = "task"+i+ " botnet botnet\n"
		f.write(s)
	f.write('input botnet botnet\n')
	f.close()


def generate_json(dag,app_path):
	f = open(app_path, 'w')
	print(app_path)
	taskname_map = dict()
	exec_profiler = dict()
	for i in dag.nodes():
		taskname_map["task"+i] = ["task"+i,True]
		exec_profiler["task"+i] = True
	final_json = {"taskname_map":taskname_map,"exec_profiler":exec_profiler}
	f.write(json.dumps(final_json,indent = 2))
	f.close()

def generate_dummy_app(dummy_conf_file,dummy_app_path):
	args = parse_args(dummy_conf_file)
	print(args)
	dummy_dag_plot = dummy_app_path + 'dag.png'
	dummy_config_path = dummy_app_path + 'configuration.txt'
	dummy_script_path = dummy_app_path + 'scripts/'
	dummy_json_path = dummy_script_path + 'config.json' 
	dummy_comm_path = dummy_script_path + 'communication.txt' 
	dummy_sample_path = dummy_app_path + 'sample_input/'
	dummy_name_path = dummy_app_path + "name_convert.txt"
	
	if os.path.isdir(dummy_app_path):
		shutil.rmtree(dummy_app_path)
	os.mkdir(dummy_app_path)

	print('Create dummy_app folder, generate DAG and plot corresponding dag.png')
	dag = prepare_task_dag(args.conf,dummy_dag_plot)

	print('Generate configuration.txt')
	generate_config(dag,dummy_config_path)

	

	os.mkdir(dummy_script_path)
	print('Generate dummy application scripts')
	generate_scripts(dag, dummy_config_path,dummy_script_path,dummy_app_path,dummy_sample_path)

	print('Generate communication.txt')
	generate_communication(dag,dummy_comm_path)

	print('Generate config.json file')
	generate_json(dag,dummy_json_path)

	print('Generate name_convert.txt')
	generate_nameconvert(dag,dummy_name_path)
	print('The dummy application is generated successfully!')

def generate_multiple_apps(dummy_app_root,dummy_conf_root,N,start,max_conf_num):
	for i in range(start,N+1):
		print(i)
		dummy_app_path= '%sdummy_app%d/'%(dummy_app_root,i)
		rand_config = random.randint(1,max_conf_num)
		dummy_conf_path = '%stask_config%d.yml'%(dummy_conf_root,rand_config)
		generate_dummy_app(dummy_conf_path, dummy_app_path)

if __name__ == '__main__':
	# generate multiple apps
	# dummy_app_root = 'dummy_app_list/'
	# dummy_conf_root = 'dummy_task_config/'
	# N = 100 # number of dags
	# start = 1
	# max_conf_num = 5
	# generate_multiple_apps(dummy_app_root,dummy_conf_root,N,start,max_conf_num)
	dummy_conf_file = 'task_config_100.yml'
	dummy_app_path = 'dummy_app_100/'
	generate_dummy_app(dummy_conf_file,dummy_app_path)
