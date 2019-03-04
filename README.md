# Automatic-DAG-Generator

Automatic DAG Generator is customized for Jupiter Orchestrator (available here: [https://github.com/ANRGUSC/Jupiter]).

## User Instructions

`rand_task_gen.py` will work with `task_config.yml` to generate a dummy application used for Jupiter. 

### task_config.yml

There are three main concepts in this file: `depth`, `width_min, width_max` and `total_num`.

`depth` parameter is used for define how many levels of the DAG, which not include the very beginning task and the last task. In other words,
If you want to create a total 6 level DAG, you should put `depth: 4`. 

`width_min` and `width_max` are used for define how many tasks in each level except the beginning and the last tasks. The number of tasks will between those two
parametes per level.

`total_num` is used for specify a total number of tasks in your DAG. For example, if you want to generate a total 10 tasks in the DAG including the beginning and the last tasks,
you should put `total_num: 8` , which 8 comes from (10 - 2). If there is a conflict between `total_num` and other parametes, the DAG will be generated with the first priority of `total_num`.

### rand_task_gen.py

To run this script, you should set `--conf task_config.yml` as the parameter. It will generate the files below:

* dummy_app/scripts -- scripts used in app_specific_files
* config.json -- used along with dummy_app/scipts
* configuration.txt -- used in CIRCE
* communication.txt -- NOT used for Jupiter, but for users to track tasks relations
* dag.png

You should also create a `sample_input/` folder and have some input files there. More details please follow the instructions on how to use Jupiter at [http://jupiter.readthedocs.io/].

### dummy_app_gen.py

This code uses the output DAG of ``rand_task_gen.py``(by Diyi), which is used to generate a random DAG, to generate the corresponding dummy application for Jupiter. To run this script, you should set `--conf task_config.yml` as the parameter. It will generate the ``dummy_app`` folder and all required content which can be used as a sample application for [Jupiter](https://github.com/ANRGUSC/Jupiter). The ``dummy_app`` has been tested and work with **Jupiter Version 3 & 4**.

## Acknowledgment
This material is based upon work supported by Defense Advanced Research Projects Agency (DARPA) under Contract No. HR001117C0053. Any views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

