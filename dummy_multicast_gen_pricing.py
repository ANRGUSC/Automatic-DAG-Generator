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

EPSILON = 1e-2

def parse_args():
	parser = argparse.ArgumentParser(description='generate random task graphs')
	parser.add_argument("--conf",required=True,type=str,help='yaml file specifying task_dag generation parameters')
	return parser.parse_args()

def random_list(depth,total_num,width_min,width_max):
    list_t = []
    if total_num>= (depth-2)*width_min+2 and total_num<=(depth-2)*width_max+2:
        list_t.append(1)
        for i in range(depth-2):
            list_t.append(2)
        list_t.append(1)
        print(list_t)
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
        while True:
            t = random.randint(width_min,width_max)
            print('-------')
            print(t)
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
	if depth==1:
		num_list = [1,total_num,1]
	else:
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
	print(edges)
	print(neighs_down_top)
	print(neighs_top_down)
	return list(set(edges)),neighs_top_down,neighs_down_top

def gen_chain_links(level_per_task):
	num_tasks = len(level_per_task)
	edges = []
	for i in range(num_tasks-1):
		edges += [(str(i),str(i+1))]
	return edges



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
		dag.nodes[str(i)]['comp'] = t
		##NOTE: actually it should not be called 'comp', but 'computation data amount'!!!
	if dag_path_plot is not None:
		plot_dag(dag,dag_path_plot)
	# print(dag.graph)
	
	return dag



def prepare_task_dag_multicast(config_yml,dag_path_plot):
	with open(config_yml) as f_config:
		config = yaml.load(f_config)
	#--- generate task graph ---

	task_per_level,level_per_task = gen_task_nodes(config['depth'],config['total_num'],config['width_min'],config['width_max'])

	if config['width_max'] > 1:
		edges,adj_list_top_down,adj_list_down_top = gen_task_links(config['deg_mu'],config['deg_sigma'],task_per_level,level_per_task)
	else:
		edges = gen_chain_links(level_per_task)
	task_comp,link_comm = gen_attr(np.arange(len(level_per_task)),edges,config['ccr'],config['comp_mu'],config['comp_sigma'],config['link_comm_sigma'])

	neighbors = {}
	multicast = {}

	for edge in edges:
		if edge[0] not in neighbors:
			neighbors[edge[0]] = [edge[1]]
		else:
			neighbors[edge[0]].append(edge[1])

	# for node in neighbors:
	# 	temp = random.uniform(0, 1)
	# 	if (temp<0.5) & (len(neighbors[node])>1) :
	# 		multicast[node] = 'true '
	# 	else:
	# 		multicast[node] = 'false '
	for node in neighbors:
		if node=='0' or len(neighbors[node])==1:
			multicast[node] = 'true '
			continue
		temp = random.uniform(0, 1)
		if (temp<0.5) & (len(neighbors[node])>1) :
			multicast[node] = 'true '
		else:
			multicast[node] = 'false '

	last_level = len(task_per_level) -1
	last_task = task_per_level[last_level][0]
	multicast[str(last_task)]='false '
	edge_d = [(e[0],e[1],{'data':link_comm[i]}) for i,e in enumerate(edges)]
	newedge_d = []
	idx = 0
	for node in neighbors:
		tmp_mc = link_comm[idx]
		for nb in neighbors[node]:
			if multicast[node]=='true ':
				newedge_d.append(tuple((node,nb,{'data':tmp_mc})))
			else:
				newedge_d.append(tuple((node,nb,{'data':link_comm[idx]})))
			idx = idx + 1

	dag = nx.DiGraph()
	dag.add_edges_from(newedge_d)
	for i,t in enumerate(task_comp):
		dag.nodes[str(i)]['comp'] = t
		##NOTE: actually it should not be called 'comp', but 'computation data amount'!!!
	if dag_path_plot is not None:
		plot_dag(dag,dag_path_plot)
	
	
	return dag,neighbors,multicast

#Generate configuration.txt
def generate_config_multitask(dag,multicast,app_path):
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
		data[i] += multicast[i]
		#data[i] += "true " ## send single output to all the children tasks
		#data[i] += "false " ## send all children output to all the children tasks correspondingly
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

def generate_communication_multicast(dag,multicast,app_path):
	f = open(app_path,'w')
	for i in dag.nodes():
		f.write("task"+i+ " "+multicast[i])
		for e0,e1,d in dag.edges(data=True):
			if i == e0:
				f.write('task'+e1+'-'+str(round(d['data'],2))+' ') #KB
		f.write("\n")
	f.close()


#Generate dummy scripts
def generate_scripts_multicast(dag,multicast,config_path,script_path,app_path,sample_path):
	

	print('------ Read input parameters from DAG  -------------')
	sorted_list = sorted(dag.nodes(data=True), key=lambda x: x[0], reverse=False)
	f = open(config_path, 'r')
	total_line = f.readlines()
	del total_line[0]
	f.close()


	for i in range(len(total_line)):
		tmp = total_line[i].split(' ')
		filename = tmp[0] + ".py"
		num = int(tmp[1])
		file_path = script_path + filename

		f = open(file_path, 'w')
		f.write("import os\n")
		f.write("import time\n")
		f.write("import shutil\n")
		f.write("import math\n")
		f.write("import random\n")

		f.write("\n")
		f.write("def task(filename,pathin,pathout):\n")
		f.write("\texecution_time = " + str(round(sorted_list[i][1]['comp'], 1)) + "\n")
		# f.write("\texecution_time\n")
		f.write("\ttimeout = time.time() + execution_time\n")
		f.write("\twhile time.time() < timeout:\n")
		f.write("\t\t1+1\n")


		f.write("\ttask_name = os.path.basename(__file__).split('.')[0]\n")
		f.write("\tprint('-------------------------')\n")
		f.write("\tprint(task_name)\n")
		f.write("\tprint(filename)\n")
		f.write("\tprint(pathin)\n")
		f.write("\tprint(pathout)\n")
		f.write("\tprint('-------------------------')\n")
		f.write("\tif isinstance(filename, list):\n")
		f.write("\t\toutput1_list = [file.split('.')[0] +'_'+task_name+'.txt' for file in filename]\n")
		f.write("\t\tinput_file = filename[0].split('_')[0]\n")
		f.write("\telif not isinstance(filename, list):\n")
		f.write("\t\toutput1_list=[filename.split('.')[0] +'_'+task_name+'.txt']\n")
		f.write("\t\tinput_file = filename.split('_')[0]\n")
		f.write("\tprint(output1_list)\n")
		f.write("\toutput1=set(output1_list)\n")
		f.write("\tprint(output1)\n")
		f.write("\tprint(input_file)\n")

		f.write("\toutput_fname=[f.split('.')[0].split('_')[-1] for f in output1]\n")
		f.write("\toutput_name='_'.join(output_fname)\n")
		f.write("\toutput_name=input_file+'_'+output_name\n")
		f.write("\tprint(output_name)\n")

		f.write("\tprint('-------------------------@@@')\n")
		f.write("\tprint(os.path.realpath('communication.txt'))\n")

		f.write("\tf = open('/centralized_scheduler/communication.txt', 'r')\n")
		f.write("\ttotal_info = f.read().splitlines()\n")
		f.write("\tf.close()\n")
		f.write("\tcomm = dict()\n")
		f.write("\tmulticast = dict()\n")
		f.write("\tfor line in total_info:\n")
		f.write("\t\tsrc = line.strip().split(' ')[0]\n")
		f.write("\t\tmulticast[src] = line.strip().split(' ')[1]\n")
		f.write("\t\tdest_info = line.split(' ')[2:-1]\n")
		f.write("\t\tif len(dest_info)>0:\n")
		f.write("\t\t\tcomm[src] = dest_info\n")
		f.write("\tprint('-------------------------##')\n")
		f.write("\tprint(multicast)\n")
		f.write("\tprint(comm)\n")
		f.write("\tprint(comm.keys())\n")
		f.write("\tprint(task_name)\n")
		f.write("\tif not os.path.isdir(pathout):\n")
		f.write("\t\tos.makedirs(pathout, exist_ok=True)\n")

		f.write("\toutput_path=[]\n")

		f.write("\tif task_name in comm.keys():\n")
		f.write("\t\tprint(comm[task_name])\n")
		f.write("\t\tdest=[x.split('-')[0] for x in comm[task_name]]\n")
		f.write("\t\tprint(dest)\n")
		f.write("\t\tcomm_data=[str(math.ceil(float(x.split('-')[1]))) for x in comm[task_name]]\n")
		f.write("\t\tprint(comm_data)\n")
		f.write("\t\toutput_list=[]\n")
		f.write("\t\tfile_size=[]\n")
		f.write("\t\tmulticast[task_name]\n")
		f.write("\t\tif multicast[task_name]=='false':\n")
		f.write("\t\t\tprint('Multicast is false')\n")
		f.write("\t\t\tfor idx,neighbor in enumerate(dest):\n")
		f.write("\t\t\t\tprint(neighbor)\n")
		f.write("\t\t\t\tprint(idx)\n")
		f.write("\t\t\t\tnew_file=output_name+'_'+neighbor\n")
		f.write("\t\t\t\toutput_list.append(new_file)\n")
		f.write("\t\t\t\tfile_size.append(comm_data[idx])\n")
		f.write("\t\t\t\tnew_path=os.path.join(pathout,new_file) \n")
		f.write("\t\t\t\toutput_path.append(new_path)\n")
		f.write("\t\t\t\tprint(new_path)\n")
		f.write("\t\t\t\tbash_script='/centralized_scheduler/generate_random_files.sh'+' '+new_path+' '+comm_data[idx]\n")
		f.write("\t\t\t\tprint(bash_script)\n")
		f.write("\t\t\t\tos.system(bash_script)\n")
		f.write("\t\telse:\n")
		f.write("\t\t\tprint('Multicast is true')\n")
		f.write("\t\t\tprint(dest[0])\n")
		f.write("\t\t\tnew_file=output_name\n")
		f.write("\t\t\tprint(comm_data)\n")
		f.write("\t\t\tprint(comm_data[0])\n")
		f.write("\t\t\tnew_path=os.path.join(pathout,new_file)\n")
		f.write("\t\t\tbash_script='/centralized_scheduler/generate_random_files.sh'+' '+new_path+' '+comm_data[0]\n")
		f.write("\t\t\tprint(bash_script)\n")
		f.write("\t\t\tos.system(bash_script)\n")
		f.write("\telif task_name not in comm.keys():\n") #final task, next node is HOME
		f.write("\t\tnew_file=output_name+'_'+task_name\n")
		f.write("\t\tnew_path=os.path.join(pathout,new_file) \n")
		f.write("\t\tprint(new_path)\n")
		f.write("\t\toutput_path.append(new_path)\n")
		f.write("\t\tfile_size=str(random.randint(1,20))\n") #random file size
		f.write("\t\tbash_script='/centralized_scheduler/generate_random_files.sh'+' '+new_path+' '+file_size\n")
		f.write("\t\tprint(bash_script)\n")
		f.write("\t\tos.system(bash_script)\n")
		
		f.write("\n")
		f.write("\treturn output_path\n")

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

if __name__ == '__main__':
	args = parse_args()
	dummy_app_path = 'dummy_complex_50/'
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
	dag,neighbors,multicast = prepare_task_dag_multicast(args.conf,dummy_dag_plot)

	print('Generate configuration.txt')
	generate_config_multitask(dag,multicast,dummy_config_path)

	os.mkdir(dummy_script_path)

	print('Generate communication.txt')
	generate_communication_multicast(dag,multicast,dummy_comm_path)

	print('Generate dummy application scripts')
	generate_scripts_multicast(dag,multicast,dummy_config_path,dummy_script_path,dummy_app_path,dummy_sample_path)

	print('Generate config.json file')
	generate_json(dag,dummy_json_path)

	print('Generate name_convert.txt')
	generate_nameconvert(dag,dummy_name_path)
	print('The dummy application is generated successfully!')