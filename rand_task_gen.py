"""
Top level config file (leave this file at the root directory). ``import config`` on the top of your file to include the global information included here.

"""
__author__ = "Diyi Hu, Jiatong Wang, Quynh Nguyen, Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2018, Autonomous Networks Research Group. All rights reserved."
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
        while True:
            t = random.randint(width_min,width_max)
            if (sum(list_t)-1+t)<total_num:
                list_t.append(t)
            elif  total_num -(sum(list_t)-1) >=width_min and total_num -(sum(list_t)-1)<=width_max :
                list_t.append(total_num -(sum(list_t)-1))
                break
            else:
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



def get_task_dag(config_yml,dag_path_plot):
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
def get_task_to_dag(dag):
	f = open("configuration.txt", 'w')
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
		data[i] += "true "
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

def get_task_to_communication(dag):
	f = open("communication.txt",'w')
	for i in dag.nodes():
		f.write("task"+i+ "\t{")
		for e0,e1,d in dag.edges(data=True):
			if i == e0:
				f.write('"task"'+e1+':"'+str(round(d['data'],2))+'KB",')
		f.write("}\n")
	f.close()

#Generate dummy scripts
def get_task_to_dummy_app():
	sorted_list = sorted(dag.nodes(data=True), key=lambda x: x[0], reverse=False)
	if "dummy_app" not in os.listdir():
		os.mkdir("dummy_app")
	f = open("configuration.txt", 'r')
	total_line = f.readlines()
	del total_line[0]
	f.close()
	for i in range(len(total_line)):
		tmp = total_line[i].split(' ')
		filename = tmp[0] + ".py"
		num = int(tmp[1])
		if num == 1:
			f = open("./dummy_app/" + filename, 'w')
			f.write("import os\n")
			f.write("import time\n")
			f.write("import shutil\n")
			f.write("import uuid\n")

			f.write("\n")
			f.write("def task(filename,pathin,pathout):\n")
			f.write("\texecution_time = " + str(round(sorted_list[i][1]['comp'], 1)) + "\n")
			f.write("\ttimeout = time.time() + execution_time\n")
			f.write("\twhile time.time() < timeout:\n")
			f.write("\t\t1+1\n")

			f.write("\tfile_name = filename.split('.')[0]\n")
			f.write("\toutput1 = file_name +'_'+str(uuid.uuid1())+'.txt'\n")
			f.write("\n")

			f.write("\tinput_path = os.path.join(pathin,filename)\n")
			f.write("\toutput_path = os.path.join(pathout,output1)\n")
			f.write("\n")
			f.write("\tshutil.copyfile(input_path,output_path)\n")
			f.write("\n")
			f.write("\treturn [output_path]\n")

			f.write("\n")
			f.write("def main():\n")
			f.write("\tfilelist = '1botnet.ipsum'\n")
			f.write("\toutpath = os.path.join(os.path.dirname(__file__), 'sample_input/')\n")
			f.write("\toutfile = task(filelist, outpath, outpath)\n")
			f.write("\treturn outfile\n")
			f.close()
		elif num > 1:
			f = open("./dummy_app/" + filename, 'w')
			f.write("import os\n")
			f.write("import time\n")
			f.write("import shutil\n")
			f.write("import uuid\n")

			f.write("\n")
			f.write("def task(filename,pathin,pathout):\n")
			f.write("\texecution_time = " + str(round(sorted_list[i][1]['comp'], 1)) + "\n")
			f.write("\ttimeout = time.time() + execution_time\n")
			f.write("\twhile time.time() < timeout:\n")
			f.write("\t\t1+1\n")

			f.write("\tfile_name = filename[0].split('.')[0]\n")
			f.write("\toutput1 = file_name +'_'+str(uuid.uuid1())+'.txt'\n")
			f.write("\n")

			f.write("\tinput_path = os.path.join(pathin,filename[0])\n")
			f.write("\toutput_path = os.path.join(pathout,output1)\n")
			f.write("\n")
			f.write("\tshutil.copyfile(input_path,output_path)\n")
			f.write("\n")
			f.write("\treturn [output_path]\n")

			f.write("\n")
			f.write("def main():\n")
			f.write("\tfilelist = ['1botnet.ipsum']\n")
			f.write("\toutpath = os.path.join(os.path.dirname(__file__), 'sample_input/')\n")
			f.write("\toutfile = task(filelist, outpath, outpath)\n")
			f.write("\treturn outfile\n")
			f.close()

# This function is not used in current version. But keep it for further uses
def get_task_to_generate_file(dag):
	if "test" not in os.listdir():
		os.mkdir("test")
	for e0,e1,d in dag.edges(data=True):
		filename = "task"+e0+"_to_"+"task"+e1+".txt"
		f=open("./test/"+filename,'w')
		s='1'*int(d['data']*1024)
		f.write(s)
		f.close()


def get_task_to_json(dag):
	f = open("config.json", 'w')
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
	dag = get_task_dag(args.conf,"dag.png")

	get_task_to_dag(dag)
	get_task_to_communication(dag)
	get_task_to_dummy_app()
	#get_task_to_generate_file(dag)
	get_task_to_json(dag)
	print('done')
