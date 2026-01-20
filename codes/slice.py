import sys
import os
import shutil
import argparse
import sentencepiece as spm
from pathlib import Path
sys.path.append("../../")
import helper
import tqdm
import random
import json
import multiprocessing 
import pickle
from cpgqls_client import *
import ast
import hashlib
import networkx as nx
import numpy as np
import datetime
import time 


cwd = os.path.dirname(os.path.realpath(__file__))
linux_dir = str(Path(cwd).parent.absolute()) + "/repos/linux"
HOME_DIR = str(Path(cwd).parent.absolute())
PROJECT_DIR = str(Path(cwd).parent.absolute())
# slices_graph_out = HOME_DIR+"/slices/slices_graph/"
joern_parse_path = PROJECT_DIR+"/joern/bin/joern-cli/joern-parse"
joern_export_path = PROJECT_DIR+"/joern/bin/joern-cli/joern-export"
def get_startlinenum(line, chosen_symbol,ignore_symbol):
    line = line.split("@@")[1]
    pieces = line.split(chosen_symbol)[1]
    try:
        if "," in pieces:
            return int(pieces.split(",")[0])
        else:
            return int(pieces.split(" ")[0])
    except ValueError:
        return int(pieces.split(" ")[0])

def line_num_patch(commit, chosen_symbol, ignore_symbol):
    cmd = "cd "+repo_dir+";git show "+commit
    # cmd = "cd "+linux_dir+";git diff --unified=0 --diff-filter=M "+commit+"^ "+commit
    result = helper.command(cmd)
    contents_start = False
    diff_start = False
    context_lines = []
    r_value = dict()
    file_paths = []
    for index in range(len(result)):
        line = result[index]
        if line.startswith("diff"):
            if len(context_lines) != 0:
                version_lines = []
                line_num = []
                for context_line in context_lines:
                    if context_line.startswith(ignore_symbol) :
                        continue
                    version_lines.append(context_line)
                for version_index in range(len(version_lines)):
                    version_line = version_lines[version_index]
                    temp_line = version_line.replace(u"\t","")
                    temp_line = temp_line.replace(" ","")
                    if temp_line.startswith(chosen_symbol+"/*") or temp_line.startswith(chosen_symbol+"*") or temp_line.startswith(chosen_symbol+"//")  or temp_line.endswith("*/\n"):
                        continue 
                    if version_line.startswith(chosen_symbol):
                        line_num.append(start_line +version_index)
                if len(line_num) != 0:
                    if file_path not in r_value.keys():
                        r_value[file_path] = []
                    r_value[file_path].extend(line_num)
                context_lines = []
            if chosen_symbol == "-":
                if "diff --cc " in line:
                    file_path = line.split("diff --cc ")[1].strip("\n")
                    # print(f"{commit} has diff -cc")
                else:
                    file_path = line.split("--git a/")[1].split(" b/")[0]
                # print(f"{file_path} {file}")
            else:
                if "diff --cc " in line:
                    # print(f"{commit} has diff -cc")
                    file_path = line.split("diff --cc ")[1].strip("\n")
                else:
                    file_path = line.split(" b/")[1].strip("\n")
            file_paths.append(file_path)
            contents_start = False
            diff_start = True
        
        if diff_start:
            if line.startswith("@@ -") :
                contents_start = True
                if len(context_lines) != 0:
                    version_lines = []
                    line_num = []
                    for context_line in context_lines:
                        if context_line.startswith(ignore_symbol) :
                            continue
                        version_lines.append(context_line)
                    for version_index in range(len(version_lines)):
                        version_line = version_lines[version_index]
                        temp_line = version_line.replace(u"\t","")
                        temp_line = temp_line.replace(" ","")
                        # if temp_line.startswith(chosen_symbol+"/*") or temp_line.startswith(chosen_symbol+"*") or temp_line.startswith(chosen_symbol+"//") or temp_line.endswith("*/\n"):
                        #     continue 
                        
                        if version_line.startswith(chosen_symbol):
                            line_num.append(start_line +version_index)
                    if len(line_num) != 0:
                        if file_path not in r_value.keys():
                            r_value[file_path] = []
                        r_value[file_path].extend(line_num)
                    context_lines = []
                 # if chosen_symbol == "-":
                #     start_line = int(line.split(",")[0].split("-")[1])
                # else:
                #     print(commit+" --- "+line)
                #     start_line = int(line.split(",")[1].split("+")[1])
                start_line = get_startlinenum(line,chosen_symbol,ignore_symbol)
            if contents_start and line.startswith("@@ -") == False:
                context_lines.append(line)
        
        if index == len(result) - 1:
            version_lines = []
            line_num = []
            for context_line in context_lines:
                if context_line.startswith(ignore_symbol) :
                    continue
                version_lines.append(context_line)
            for version_index in range(len(version_lines)):
                version_line = version_lines[version_index]
                temp_line = version_line.replace(" ","")
                temp_line = temp_line.replace("\t","")
                # if temp_line.startswith(chosen_symbol+"/*") or temp_line.startswith(chosen_symbol+"*") or temp_line.startswith(chosen_symbol+"//") or temp_line.endswith("*/"):
                #     continue 
                if version_line.startswith(chosen_symbol):
                    line_num.append(start_line + version_index)
            if len(line_num) != 0:
                if file_path not in r_value.keys():
                    r_value[file_path] = []
                
                r_value[file_path].extend(line_num)
                
    return r_value,file_paths

# def joern_slice_graph(commit):
#     helper.create_dir(slices_graph_out+commit)
#     for index in range(len(symbols)):
#         symbol = symbols[index]
#         src_dir = depends_out+commit+"/"+out_dirs[index]+"/"
#         file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
#         helper.create_dir(slices_graph_out+commit+"/"+out_dirs[index])
#         if os.path.exists(src_dir) == False:
#             continue
#         for file in os.listdir(src_dir):
#             contents = helper.readFile(src_dir+file)
#             edges = ast.literal_eval(contents[0])
#             graph = create_adjacency_list(edges)
#             out_path = slices_graph_out+commit+"/"+out_dirs[index]+"/"+file
#             helper.create_dir_if_not_exist(out_path)
#             file = file.replace("$","/")
#             line_nums = file_linenum[file]
#             backwards = set()
#             forwards = set()
#             for line_num in line_nums:
#                 backwards.update(create_backward_slice(graph,line_num)) 
#                 forwards.update(create_forward_slice(graph,line_num))
#             backwards = list(backwards)
#             forwards = list(forwards)
#             backwards.sort()
#             forwards.sort()
#             for line_num in backwards:
#                 if line_num not in line_nums:
#                     helper.dump(out_path+"/backward",str(line_num)+"\n")
#             for line_num in forwards:
#                 if line_num not in line_nums:
#                     helper.dump(out_path+"/forward",str(line_num)+"\n")
      
symbols = ["-","+"]
status_symbols = ["^",""]  
out_dirs = ["before_patch","after_patch"]  


   
def joern_slice(commit):
    cwd_dir = os.getcwd() + '/slices'
    work_dir = cwd_dir + f"/{commit}/"

    # if os.path.exists(work_dir):
    #     print(f"dumplicate work dir: {work_dir} commit id is not even unique! Must be something wrong")
    #     exit()
  
    helper.create_dir(work_dir)
    helper.create_dir(depends_edges+commit)

    temp_c = work_dir+"temp.c"
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        file_paths = [*set(file_paths)]
        
        helper.create_dir_if_not_exist(depends_edges+commit+"/"+out_dirs[index])
        # helper.create_dir_if_not_exist(slices_out+commit+"/"+out_dirs[index])
        for file_path in file_paths:
            if file_path not in file_linenum:
                continue
            ret = os.system("cd "+repo_dir+";git show "+commit+status_symbols[index]+":"+file_path+" > "+temp_c)
            if ret != 0:
                print(f"{commit} {index} {file_path} git show failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            line_nums = file_linenum[file_path]
            file_path1 = file_path.replace("/","$")
            depends_edges_path = depends_edges+commit+"/"+out_dirs[index]+"/"+file_path1
            # slices_out_path = slices_out+commit+"/"+out_dirs[index]+"/"+file_path1
        
            cpg_out = work_dir+"cpg.bin"
            ret = os.system(f'rm -rf {cpg_out}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern cpg remove failed before parsing!")
    
            #ret = os.system(f'sh {joern_parse_path} {temp_c} -o {cpg_out} >/dev/null 2>&1')
            ret = os.system(f'sh {joern_parse_path} {temp_c} -o {cpg_out} > /tmp/joern_parse.log 2>&1')
            if ret != 0:
                print(f'sh {joern_parse_path} {temp_c} -o {cpg_out} >/dev/null 2>&1')
                print(f"{commit} {index} {file_path} joern parsing failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            dot_dir = work_dir+"dot/"
            ret = os.system(f'rm -rf {dot_dir}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern dot remove failed before exporting!")
            ret = os.system(f'sh {joern_export_path} {cpg_out} --repr pdg -o {dot_dir} >/dev/null 2>&1')
            if ret != 0:
                print(f'sh {joern_export_path} {cpg_out} --repr pdg -o {dot_dir} >/dev/null 2>&1')
                print(f"{commit} {index} {file_path} joern export failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            dot_nums = len(os.listdir(dot_dir))
            depends = []
            node_line_dict = dict()
            for num in range(dot_nums):
                dot_path = dot_dir+str(num)+"-pdg.dot"
                dotpdg_str = helper.readFile(dot_path)
                for line in dotpdg_str:
                    if "[label = " in line and "label = <(METHOD_RETURN" not in line and "<SUB>" in line and "</SUB>" in line:
                        node=line.split("\" [label = ")[0].split("\"")[1]
                        line = line.split("<SUB>")[1].split("</SUB>")[0]
                        node_line_dict[node] = line
                    if "\" -> \"" in line:
                        in_node = line.split("\" -> \"")[0].split("\"")[1]
                        out_node = line.split("\" -> \"")[1].split("\"")[0]
                        if in_node in node_line_dict and out_node in node_line_dict:
                            in_line = int(node_line_dict[in_node])
                            out_line = int(node_line_dict[out_node])
                            edge = ""
                            if "label =" in line:
                                if "CDG" in line:
                                    edge = "CDG"
                                if "DDG" in line:
                                    edge = "DDG"
                            depends.append((in_line,out_line,edge))
          
            ret = os.system(f'rm -rf {cpg_out}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern cpg remove failed!")
            ret = os.system(f'rm -rf {dot_dir}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern dot remove failed!")
            
            helper.dump(depends_edges_path,str(depends))
           
            # graph = create_adjacency_list(depends)
            # backwards = set()
            # forwards = set()
            # for line_num in line_nums:
            #     backwards.update(create_backward_slice(graph,line_num)) 
            #     forwards.update(create_forward_slice(graph,line_num))
            # backwards = list(backwards)
            # forwards = list(forwards)
            # backwards.sort()
            # forwards.sort()
            # helper.create_dir_if_not_exist(slices_out_path)
            # backward_path = slices_out_path + "/backward"
            # forward_path = slices_out_path + "/forward"
            # for line_num in backwards:
            #     if line_num not in line_nums:
            #         helper.dump(backward_path,str(line_num)+"\n")
            # for line_num in forwards:
            #     if line_num not in line_nums:
            #         helper.dump(forward_path,str(line_num)+"\n")
                
    ret = os.system(f'rm -rf {work_dir}')
    if ret != 0:
        print(f"{work_dir} remove failed!")


def joern_slice_label(commit):
    
    cwd_dir = os.getcwd() + '/slice'
    work_dir = cwd_dir + f"/{commit}/"
    # if os.path.exists(work_dir):
    #     print(f"dumplicate work dir: {work_dir} commit id is not even unique! Must be something wrong")
    #     exit()
    
    helper.create_dir(work_dir)
    helper.create_dir(depends_edges+commit)
    # helper.create_dir(slices_out+commit)
    temp_c = work_dir+"temp.c"
    
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        file_paths = [*set(file_paths)]
        
        helper.create_dir_if_not_exist(depends_edges+commit+"/"+out_dirs[index])
        # helper.create_dir_if_not_exist(slices_out+commit+"/"+out_dirs[index])
        for file_path in file_paths:
            if file_path not in file_linenum:
                continue
            ret = os.system("cd "+repo_dir+";git show "+commit+status_symbols[index]+":"+file_path+" > "+temp_c)
            if ret != 0:
                print(f"{commit} {index} {file_path} git show failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            line_nums = file_linenum[file_path]
            file_path1 = file_path.replace("/","$")
            depends_edges_path = depends_edges+commit+"/"+out_dirs[index]+"/"+file_path1
            # slices_out_path = slices_out+commit+"/"+out_dirs[index]+"/"+file_path1
            
            cpg_out = work_dir+"cpg.bin"
            ret = os.system(f'rm -rf {cpg_out}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern cpg remove failed before parsing!")
    
            ret = os.system(f'sh {joern_parse_path} {temp_c} -o {cpg_out} >/dev/null 2>&1')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern parsing failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            dot_dir = work_dir+"dot/"
            ret = os.system(f'rm -rf {dot_dir}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern dot remove failed before exporting!")
            ret = os.system(f'sh {joern_export_path} {cpg_out} --repr ddg -o {dot_dir} >/dev/null 2>&1')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern export failed!")
                shutil.rmtree(work_dir, ignore_errors=True)
                return
            dot_nums = len(os.listdir(dot_dir))
            depends = []
            node_line_dict = dict()
            for num in range(dot_nums):
                dot_path = dot_dir+str(num)+"-ddg.dot"
                dotpdg_str = helper.readFile(dot_path)
                for line in dotpdg_str:
                    if "[label = " in line and "label = <(METHOD_RETURN" not in line and "<SUB>" in line and "</SUB>" in line:
                        node=line.split("\" [label = ")[0].split("\"")[1]
                        line = line.split("<SUB>")[1].split("</SUB>")[0]
                        node_line_dict[node] = line
                    if "\" -> \"" in line:
                        in_node = line.split("\" -> \"")[0].split("\"")[1]
                        out_node = line.split("\" -> \"")[1].split("\"")[0]
                        if in_node in node_line_dict and out_node in node_line_dict:
                            in_line = int(node_line_dict[in_node])
                            out_line = int(node_line_dict[out_node])
                            edge = ""
                            if "label =" in line:
                                edge = line.split("label = \"")[1].split("\"")[0]
                            depends.append((in_line,out_line,edge))
          
            ret = os.system(f'rm -rf {cpg_out}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern cpg remove failed!")
            ret = os.system(f'rm -rf {dot_dir}')
            if ret != 0:
                print(f"{commit} {index} {file_path} joern dot remove failed!")
            
            helper.dump(depends_edges_path,str(depends))

                
    ret = os.system(f'rm -rf {work_dir}')
    if ret != 0:
        print(f"{work_dir} remove failed!")
    done_commit_path = done_out+commit    
    os.system(f'mkdir {done_commit_path}')


def create_adjacency_list( edges):
    adjacency_list = {}
    for edge in edges:
        start_ln = edge[0]
        end_ln = edge[1]
        if start_ln not in adjacency_list:
            adjacency_list[start_ln] = set()
        adjacency_list[start_ln].add(end_ln)
    return adjacency_list

def create_forward_slice(adjacency_list, line_no):
    sliced_lines =set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append(line_no)
    while len(stack) != 0:
        cur = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        if cur not in adjacency_list:
            continue
        adjacents = adjacency_list[cur]
        for node in adjacents:
            if node not in sliced_lines:
                stack.append(node)
    return sliced_lines

def create_adjacency_list1( edges):
    adjacency_list = {}
    for edge in edges:
        start_ln = edge[0]
        end_ln = edge[1]
        label = edge[2]
        if start_ln not in adjacency_list:
            adjacency_list[start_ln] = set()
        adjacency_list[start_ln].add((end_ln,label))
    return adjacency_list

def create_forward_slice1(adjacency_list, line_no):
    sliced_lines =set()
    sliced_lines.add(line_no)
    stack = list()
    stack.append((line_no,None))
    while len(stack) != 0:
        cur, prev_label = stack.pop()
        if cur not in sliced_lines:
            sliced_lines.add(cur)
        if cur not in adjacency_list:
            continue
        adjacents = adjacency_list[cur]
        for node, label in adjacents:
            if node not in sliced_lines:
                if prev_label is None or prev_label == label:
                    stack.append((node,label))
    return sliced_lines

def invert_graph(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node in adj:
            if node not in igraph:
                igraph[node] = set()
            igraph[node].add(ln)
    return igraph

def invert_graph1(adjacency_list):
    igraph = {}
    for ln in adjacency_list.keys():
        igraph[ln] = set()
    for ln in adjacency_list:
        adj = adjacency_list[ln]
        for node, label in adj:
            if node not in igraph:
                igraph[node] = set()
            igraph[node].add((ln,label))
    return igraph
    
def create_backward_slice(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph(adjacency_list)
    return create_forward_slice(inverted_adjacency_list, line_no)

def create_backward_slice1(adjacency_list, line_no):
    inverted_adjacency_list = invert_graph1(adjacency_list)
    return create_forward_slice1(inverted_adjacency_list, line_no)

def dfs(node, visited, adjacency_list):
    if node in visited:
        return
    visited.add(node)
    if node not in adjacency_list:
        path_depths.append(len(visited))
        print(path_depths)
        return
    adjacents = adjacency_list[node]
    for adj in adjacents:
        dfs(adj, visited, adjacency_list)


def ddg_estimate_thread(commit):
    commit_path = depends_out+commit
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        if os.path.exists(commit_path+"/"+out_dirs[index]) == False:
            return -1,-1
    backward_distances = []
    forward_distances = []
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        for file in os.listdir(commit_path+"/"+out_dirs[index]):
            file_path = commit_path+"/"+out_dirs[index]+"/"+file
            depends = helper.readFile(file_path)
            edges = []
            if depends[0] == "":
                print("depends[0] is null")
            try:
                depends = ast.literal_eval(depends[0])
            except SyntaxError:
                print(f"commit: {commit}")
                print("depends[0]: ",depends[0])
                os._exit(1) 
            for depend in depends:
                if depend[2] == "DDG":
                    edges.append((depend[0],depend[1]))
                    
            graph = create_adjacency_list(edges)
            file_path1 = file_path.split("/")[-1].replace("$","/")
            try:
                line_nums = file_linenum[file_path1]
            except KeyError  as e:
                print("key error: ",commit)
                return -1,-1
            
            backwards = set()
            forwards = set()
            for line_num in line_nums:
                for backward_line in create_backward_slice(graph,line_num):
                    if backward_line not in line_nums:
                        backward_distance = abs(int(line_num)-int(backward_line))
                        if backward_distance != 0:
                            backward_distances.append(backward_distance)
                    
                for forward_line in create_forward_slice(graph,line_num):
                    if forward_line not in line_nums:
                        forward_distance = abs(int(forward_line)-int(line_num))
                        if forward_distance != 0:
                            forward_distances.append(forward_distance)
                    
    return backward_distances,forward_distances


def ddg_estimate():
    global depends_out
    dir_path = "1M/"
    depends_out = HOME_DIR+"/slices/"+dir_path+"depends_edges/"
    # commits = ["44039e00171b","9077f052abd","e6193f78bb68"]
    done_commits = []
    commits = os.listdir(depends_out)
    print("to do commits: ",len(commits))
    distances = [[],[]]
    types = ["backward","forward"]
    count = 0
    not_valid = 0
    print(datetime.datetime.now().time())
    with multiprocessing.Pool(60) as pool:
        for i in pool.imap(ddg_estimate_thread,commits):
            count += 1
            if i[0] == -1 and i[1] == -1:
                not_valid += 1
                continue
            distances[0].extend(i[0])
            distances[1].extend(i[1])
            if count % 10000 == 0:
                print(datetime.datetime.now().time())
                print(f"processed {count} commits")
                print("not valid: ",not_valid)
                for index in range(len(types)):
                    print("{}, mean ddg distance: {}".format(types[index],np.mean(np.array(distances[index]))))
                    print("{}, middle ddg distance: {}".format(types[index],np.median(np.array(distances[index]))))
                    print("0%% path length: {}".format(np.percentile(np.array(distances[index]),0)))
                    print("25%% path length: {}".format(np.percentile(np.array(distances[index]),25)))
                    print("75%% path length: {}".format(np.percentile(np.array(distances[index]),75)))
                    print("100%% path length: {}".format(np.percentile(np.array(distances[index]),100)))
                
    distances = np.array(distances)
    print("not valid: ",not_valid)
    for index in range(len(types)):
        print("type: ",types[index])
        print("# of relationships: ", len(distances[index]))
        print("mean ddg distance: {}".format(np.mean(distances[index])))
        print("middle ddg distance: {}".format(np.median(distances[index])))
        print("0%% path length: {}".format(np.percentile(distances[index],0)))
        print("25%% path length: {}".format(np.percentile(distances[index],25)))
        print("75%% path length: {}".format(np.percentile(distances[index],75)))
        print("100%% path length: {}".format(np.percentile(distances[index],100)))

    
#a0f52c3d3
def ddg_thread(commit):
    commit_path = depends_out+commit
    backward_distance = distance
    forward_distance = distance
    
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        if os.path.exists(commit_path+"/"+out_dirs[index]) == False:
            joern_slice(commit)
            # print("fdsfds")
            # continue
            # return -1,commit
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        for file in os.listdir(commit_path+"/"+out_dirs[index]):
            file_path = commit_path+"/"+out_dirs[index]+"/"+file
            depends = helper.readFile(file_path)
            edges = []
            if depends[0] == "":
                print("depends[0] is null")
            try:
                depends = ast.literal_eval(depends[0])
            except SyntaxError:
                print(f"commit: {commit}")
                print("depends[0]: ",depends[0])
                return -1,commit
            except ValueError:
                print(f"commit: {commit}")
                print("depends[0]: ",depends[0])
                return -1,commit
            for depend in depends:
                if depend[2] == "DDG":
                    edges.append((depend[0],depend[1]))
                    
            graph = create_adjacency_list(edges)
            backwards = set()
            forwards = set()
            file_path1 = file_path.split("/")[-1].replace("$","/")
            line_nums = file_linenum[file_path1]
            for line_num in line_nums:
                if distance == 0:
                    backwards.update(create_backward_slice(graph,line_num)) 
                    forwards.update(create_forward_slice(graph,line_num))
                else:
                    for backward_line in create_backward_slice(graph,line_num):
                        if int(line_num) - int(backward_line) <= backward_distance:
                            backwards.add(backward_line)
                    for forward_line in create_forward_slice(graph,line_num):
                        if int(forward_line) - int(line_num) <= forward_distance:
                            forwards.add(forward_line)
            
            backwards = list(backwards)
            forwards = list(forwards)
            backwards.sort()
            forwards.sort()
            ddgs_out_path = ddgs_out+commit+"/"+out_dirs[index]+"/"+file_path.split("/")[-1]
            helper.create_dir_if_not_exist(ddgs_out_path)
            backward_path = ddgs_out_path + "/backward"
            forward_path = ddgs_out_path + "/forward"
            helper.delFileIfExists(backward_path)
            helper.delFileIfExists(forward_path)
            
            for line_num in backwards:
                if line_num not in line_nums:
                    helper.dump(backward_path,str(line_num)+"\n")
            for line_num in forwards:
                if line_num not in line_nums:
                    helper.dump(forward_path,str(line_num)+"\n")
    return 1,commit


def direct_ddg_thread(commit):
    commit_path = depends_out+commit
    
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        if os.path.exists(commit_path+"/"+out_dirs[index]) == False:
            joern_slice_label(commit)
            continue
            
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
      
        for file in os.listdir(commit_path+"/"+out_dirs[index]):
            file_path = commit_path+"/"+out_dirs[index]+"/"+file
            depends = helper.readFile(file_path)
            edges = []
            if depends[0] == "":
                print("depends[0] is null")
            try:
                depends = ast.literal_eval(depends[0])
            except SyntaxError:
                print(f"commit: {commit}")
                print("depends[0]: ",depends[0])
                os._exit(1) 
            for depend in depends:
                edges.append((depend[0],depend[1],depend[2]))
                    
            graph = create_adjacency_list1(edges)
            backwards = set()
            forwards = set()
            file_path1 = file_path.split("/")[-1].replace("$","/")
            line_nums = file_linenum[file_path1]
            for line_num in line_nums:
                backwards.update(create_backward_slice1(graph,line_num)) 
                forwards.update(create_forward_slice1(graph,line_num))
                
            backwards = list(backwards)
            forwards = list(forwards)
            backwards.sort()
            forwards.sort()
            ddgs_out_path = ddgs_out+commit+"/"+out_dirs[index]+"/"+file_path.split("/")[-1]
            helper.create_dir_if_not_exist(ddgs_out_path)
            backward_path = ddgs_out_path + "/backward"
            forward_path = ddgs_out_path + "/forward"
            helper.delFileIfExists(backward_path)
            helper.delFileIfExists(forward_path)
            #print(commit, line_nums)
            for line_num in backwards:
                if line_num not in line_nums:
                    helper.dump(backward_path,str(line_num)+"\n")
            for line_num in forwards:
                if line_num not in line_nums:
                    helper.dump(forward_path,str(line_num)+"\n")
    return 1,commit

def ddg(distance1):
    global distance
    distance = int(distance1)
    global depends_out
    dir_path = "finetune/"
    HOME_DIR = "/data1/xli399/old_stuff"
    depends_out = HOME_DIR+"/slices/"+dir_path+"depends_edges/"    
    global depends_edges
    depends_edges = depends_out    
    global ddgs_out
    if distance == 0:
        ddgs_out = HOME_DIR+"/slices/"+dir_path+"slices_onlydata/"
    else:
        ddgs_out = HOME_DIR+"/slices/"+dir_path+"slices_onlydata_distance_"+str(distance)+"/"
    helper.create_dir_if_not_exist(ddgs_out)
    ddg_thread("8318d78a44d4")
    return
    commits = os.listdir(depends_out)
    # done_commits = os.listdir(ddgs_out)
    # # todo_commits = ["44039e00171b","9077f052abd5","e6193f78bb68"]
    # todo_commits = [x for x in commits if x not in done_commits]
    print("to do commits: ",len(commits))
    count = 0
    invalid = 0
    with multiprocessing.Pool(20) as pool:
        for i in pool.imap(ddg_thread,commits):
            count += 1
            print(count)
            if i[0] == -1:
                invalid += 1
            if count % 10000 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))
    #invalid commits: ['3ff901cb5df1', 'f01cfbaf9b29', '5b102782c7f4', 'ae962e5f630f', 'e548f749b096', '8ed37e791960', 'd0a9123ef548', '046d2e7c50e3', '453a2a82050e', '43cd72c5892c', '022143d0c52b', '1f38b8c564b8', '8e280369b907']


def direct_ddg():
    dir_path = "finetune/"
    global depends_out
    global depends_edges
    depends_out = HOME_DIR+"/slices/"+dir_path+"depends_edges_labels/"  
    depends_edges = depends_out      
    global ddgs_out
    ddgs_out = HOME_DIR+"/slices/"+dir_path+"slices_directdd/"
    helper.create_dir_if_not_exist(ddgs_out)
    commits = os.listdir(depends_out)
    done_commits = os.listdir(ddgs_out)
    todo_commits = commits
    print("to do commits: ",len(todo_commits))
    count = 0
    invalid = []
    with multiprocessing.Pool(60) as pool:
        for i in pool.imap(direct_ddg_thread,todo_commits):
            count += 1
            if i[0] == -1:
                invalid.append(i[1])
            if count % 1000 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))


def tune_thread(commit):
    commit_path = depends_edges+commit
    
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        if os.path.exists(commit_path+"/"+out_dirs[index]) == False:
            # joern_slice(commit)
            # continue
            return -1,commit
    for index in range(len(symbols)):
        file_linenum,file_paths = line_num_patch(commit, symbols[index], symbols[1-index])
        # print(file_paths)
        # print(os.listdir(commit_path+"/"+out_dirs[index]))
        for file in os.listdir(commit_path+"/"+out_dirs[index]):
            file_path = commit_path+"/"+out_dirs[index]+"/"+file
            depends = helper.readFile(file_path)
            edges = []
            if depends[0] == "":
                print("depends[0] is null")
            try:
                depends = ast.literal_eval(depends[0])
            except SyntaxError:
                print(f"commit: {commit}")
                print("depends[0]: ",depends[0])
                os._exit(1) 
            for depend in depends:
                if depend[2] == "DDG":
                    edges.append((depend[0],depend[1]))
                    
            graph = create_adjacency_list(edges)
            backwards = set()
            forwards = set()
            file_path1 = file_path.split("/")[-1].replace("$","/")
            line_nums = file_linenum[file_path1]
            for line_num in line_nums:
                forwards.update(create_forward_slice(graph,line_num))
            graph = create_adjacency_list(depends)
            for line_num in line_nums:
                backwards.update(create_backward_slice(graph,line_num)) 
            
            backwards = list(backwards)
            forwards = list(forwards)
            backwards.sort()
            forwards.sort()
            tunes_out_path = tunes_out+commit+"/"+out_dirs[index]+"/"+file_path.split("/")[-1]
            helper.create_dir_if_not_exist(tunes_out_path)
            backward_path = tunes_out_path + "/backward"
            forward_path = tunes_out_path + "/forward"
            helper.delFileIfExists(backward_path)
            helper.delFileIfExists(forward_path)
            
            for line_num in backwards:
                if line_num not in line_nums:
                    helper.dump(backward_path,str(line_num)+"\n")
            for line_num in forwards:
                if line_num not in line_nums:
                    helper.dump(forward_path,str(line_num)+"\n")
    return 1,commit

def tune(source):
    dir_path = source+"/"
    global depends_edges
    depends_edges = HOME_DIR+"/slices/"+dir_path+"depends_edges/"        
    global tunes_out    
    tunes_out = HOME_DIR+"/slices/"+dir_path+"slice_tunes/"
    helper.create_dir_if_not_exist(tunes_out)
    
    commits = os.listdir(depends_edges)
    todo_commits = commits
    print("to do commits: ",len(todo_commits))
    count = 0
    invalid = []
    with multiprocessing.Pool(60) as pool:
        for i in pool.imap(tune_thread,todo_commits):
            count += 1
            if i[0] == -1:
                invalid.append(i[1])
            if count % 100000 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))

def slice(dir_path,joern_label):
    start_time = time.time()
    global depends_out
    global depends_edges
    global done_out
    done_out = HOME_DIR+"/slices/"+dir_path+"done_edges_labels/"
    depends_out = HOME_DIR+"/slices/"+dir_path+"depends_edges_labels/"
    depends_edges = depends_out
    # depends_out = HOME_DIR+"/slices/"+dir_path+"depends/"
    
    
    todo_commits = []
    done_commits = []
    for file in os.listdir(done_out):
        done_commits.append(file)
    src_file = HOME_DIR+"/slices/"+dir_path+"commits"
    lines = helper.readFile(src_file)
    for line in lines:
        commit = line.strip("\n")
        if commit.startswith("a") or commit.startswith("b") or commit.startswith("c") or commit.startswith("d"):
            if commit not in done_commits:
                todo_commits.append(line.strip("\n"))
   
    print("all: {}, to do: {}".format(len(lines),len(todo_commits)))
    if joern_label:
        slice_method = joern_slice_label
    else:
        slice_method = joern_slice
    with multiprocessing.Pool(20) as pool:
        counter = 0
        for i in pool.imap(slice_method,todo_commits):
            counter += 1
            if counter % 10000 == 0:
                print("counter: {}".format(counter))
                print("duration since start: {}".format(time.time()-start_time))

def cve_slice_ddg(method):
    global repo_dir
    repo_dir = linux_dir
    # repo_dir = "/home/xli399/Projects/SlicingBert/repos/FFmpeg"
    proj = "treevul"
    src_file = HOME_DIR+"/data/cves/treevul"
    lines = helper.readFile(src_file)
    commits = []
    for line in lines:
        commits.append(line.strip("\n"))
    # src_file = HOME_DIR+"/data/cves/ffmpeg_other_eval"
    # lines = helper.readFile(src_file)
    # for line in lines:
    #     commits.append(line.split(":")[0])
    
    global depends_out
    global depends_edges
    depends_out = HOME_DIR+"/slices/"+proj+"/depends_edges/"
    helper.create_dir_if_not_exist(depends_out)
    depends_edges = depends_out
    # print("slicing for CVE commits starts")
    # commits.remove("578b956fe741")
   
    print(repo_dir)
    # commits.remove("ae50d8270026")
    with multiprocessing.Pool(60) as pool:
        counter = 0
        for i in pool.imap(joern_slice,commits):
            counter += 1
            if counter % 100 == 0:
                print("counter: {}".format(counter))
    if method == 0:
        global ddgs_out
        ddgs_out = HOME_DIR+"/slices/"+proj+"/slices_onlydata/"
        helper.create_dir_if_not_exist(ddgs_out)
        global distance
        distance = 0
        called_func = ddg_thread
    else:
        global tunes_out    
        tunes_out = HOME_DIR+"/slices/"+proj+"/slice_tunes/"
        helper.create_dir_if_not_exist(tunes_out)
        called_func = tune_thread
    commits = os.listdir(depends_out)
    print("Generate ddg for CVE commits starts")
    count = 0
    invalid = []
  
    with multiprocessing.Pool(30) as pool:
        for i in pool.imap(called_func,commits):
            count += 1
            if i[0] == -1:
                invalid.append(i[1])
            if count % 100 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))
    
def given_ddg():
    # commits = [x.strip("\n") for x in helper.readFile("/home/xli399/Projects/SlicingBert/main/results/evaluation/random_passed_cases/pred_mem_actual_non_mem/difficult")]
    commits = [x.strip("\n") for x in helper.readFile("/home/xli399/Projects/SlicingBert/main/results/evaluation/random_passed_cases/updated_groundtruth/difficult_commits")]
    
    commits += [x.strip("\n") for x in helper.readFile("/home/xli399/Projects/SlicingBert/main/results/evaluation/random_passed_cases/updated_groundtruth/easy_commits")]
    global depends_out
    global depends_edges
    depends_out = HOME_DIR+"/slices/given/updated_groundtruth/depends_edges/"
    helper.create_dir_if_not_exist(depends_out)
    depends_edges = depends_out

    global ddgs_out
    ddgs_out = HOME_DIR+"/slices/given/updated_groundtruth/slices_onlydata/"
    helper.create_dir_if_not_exist(ddgs_out)
    global distance
    distance = 0

    # commits = os.listdir(depends_out)
    print("Generate ddg for CVE commits starts")
    count = 0
    invalid = []
    with multiprocessing.Pool(30) as pool:
        for i in pool.imap(ddg_thread,commits):
            count += 1
            print(count)
            if i[0] == -1:
                invalid.append(i[1])
            if count % 100 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))
#commits: to do commits    
#name: name of the evalautin data, and should be same with the name argument of build_eval_data_for_random_given in preprocess.py
#repo_dir: the repo dir of the commits, for example the linux git repo
def given_slice(commits,name,repo_dir1):
    # commits = [x.strip("\n") for x in helper.readFile("/home/xli399/Projects/SlicingBert/main/results/evaluation/random_passed_cases/pred_mem_actual_non_mem/difficult")]
    global repo_dir
    repo_dir = repo_dir1
    global depends_out
    global depends_edges
    depends_out = HOME_DIR+"/slices/"+name+"/depends_edges/"
    helper.create_dir_if_not_exist(depends_out)
    depends_edges = depends_out

    global ddgs_out
    ddgs_out = HOME_DIR+"/slices/"+name+"/slices_onlydata/"
    helper.create_dir_if_not_exist(ddgs_out)
    global distance
    distance = 0

    # commits = os.listdir(depends_out)
    print("Generate ddg for "+name+" commits starts")
    count = 0
    invalid = []
    with multiprocessing.Pool(30) as pool:
        for i in pool.imap(ddg_thread,commits):
            count += 1
            print(count)
            if i[0] == -1:
                invalid.append(i[1])
            if count % 100 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))
    
 
 