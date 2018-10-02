__author__ = "Diyi Hu, Jiatong Wang, Bhaskar Krishnamachari"
__copyright__ = "Copyright (c) 2018, Autonomous Networks Research Group. All rights reserved."
__license__ = "GPL"
__version__ = "1.0"

#import scipy.sparse
import argparse
import numpy as np
import random
from functools import reduce
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
#from networkx.drawing.layout import rescale_layout
import pylab as plt
import yaml
import os

EPSILON = 1e-2

def parse_args():
	parser = argparse.ArgumentParser(description='generate random task graphs')
	parser.add_argument("--conf",required=True,type=str,help='yaml file specifying task_dag generation parameters')
	return parser.parse_args()



def gen_task_nodes(depth,width_min,width_max):
	num_levels = depth+2		# 2: 1 for entry task, 1 for exit task
	num_nodes_per_level = np.array([random.randint(width_min,width_max) for i in range(num_levels)])
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

	task_per_level,level_per_task = gen_task_nodes(config['depth'],config['width_min'],config['width_max'])
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

#========================================================================================
def get_task_to_dag(dag):
	f = open("Dag.txt",'w')
	total_node = len(dag.nodes())
	f.write(str(total_node)+"\n")
	task_dict ={}
	for j in dag.nodes():
		task_dict[j] = 0
	for e0,e1 in dag.edges():
		if e1 in task_dict.keys():
			task_dict[e1] +=1
	
	
	for i in dag.nodes():
		f.write("task"+i+ " ")
		f.write(str(task_dict[i])+" ")
		f.write("true ")
		for e0,e1 in dag.edges():
			if i == e0:
				f.write("task"+e1+" ")
		if int(i) == total_node - 1:
			f.write("home")

		f.write("\n")

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
	
def get_task_to_dummy_app(dag):
	if "dummy_app" not in os.listdir():
		os.mkdir("dummy_app")
	for n,d in dag.nodes(data=True):
		filename = "task"+n+".py"
		f=open("./dummy_app/"+filename,'w')
		f.write("import time\n")
		f.write("execution_time = "+str(round(d['comp'],1))+"\n")
		f.write("timeout = time.time() + execution_time\n")
		f.write("while time.time() < timeout:\n")
		f.write("\t1+1")
		f.close()

		
def get_task_to_generate_file(dag):
	if "test" not in os.listdir():
		os.mkdir("test")
	for e0,e1,d in dag.edges(data=True):
		filename = "task"+e0+"_to_"+"task"+e1+".txt"
		f=open("./test/"+filename,'w')
		s='1'*int(d['data']*1024)
		f.write(s)
		f.close()

#===================================================================================

if __name__ == '__main__':
	args = parse_args()
	#dag_path_plot="D:/"
	dag = get_task_dag(args.conf,"dag.png")
	#print(dag.node['0'])
	#for e0,e1,d in dag.edges(data=True):
		#print(e0,e1,d)
	#nx.draw(dag)
	#plt.savefig("test.png")
	#plt.show()
	#plot_dag(dag)
	get_task_to_dag(dag)	# question1
	get_task_to_communication(dag) # question2
	get_task_to_dummy_app(dag) # question3
	get_task_to_generate_file(dag) # question4
	print('done')
