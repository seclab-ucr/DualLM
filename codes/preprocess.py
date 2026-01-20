import sys,os
import argparse
import sentencepiece as spm
from pathlib import Path
sys.path.append("../../")
import helper
import tqdm
import random
import json
from multiprocessing import Pool
import multiprocessing 
from filelock import FileLock
from multiprocessing import Queue
import subprocess
import ast
import shutil
import time 
import glob

cwd = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = str(Path(cwd).parent.absolute())
linux_dir = str(Path(cwd).parent.absolute()) + "/repos/linux"
HOME_DIR = str(Path(cwd).parent.absolute())

  
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
        # print(line)
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
                    #print(f"{commit} has diff -cc")
                else:
                    file_path = line.split("--git a/")[1].split(" b/")[0]
                # print(f"{file_path} {file}")
            else:
                if "diff --cc " in line:
                    #print(f"{commit} has diff -cc")
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
                if line.startswith("\ No newline at end of file") == False:
                    context_lines.append(line)
                # context_lines.append(line)
        
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

wrong_slcings_commits = []
     

def single_thread(commit):
    counters = [0,0]
    for i in range(len(statuses)):
        status = statuses[i]
        backward_symbol = backward_symbols[i]
        patch_symbol = patch_symbols[i]
        forward_symbol = forward_symbols[i]
        
        out_dir = raw_data_dir + status + "/"
        
        lines =[None]*3
        # print(commit)
        file_linenum,filename_path = line_num_patch(commit, symbols[i], symbols[1-i])
        #print(commit,file_linenum)
        if file_linenum != {}:
            counters[i] += 1
        
        for patch_file_name in file_linenum.keys():
            forward_slicing = []
            backward_slicing = []
            patch_lines = []
            cmd = "cd "+repo_dir+";git show "+commit+commit_status_symbols[i]+":"+patch_file_name
            file_lines1 = helper.command(cmd)
            #print(commit,file_lines1)
            file_lines = []
            for line in file_lines1:
                file_lines.append(line.replace("\t","").strip("\n"))
            line_nums = file_linenum[patch_file_name]
            for num in line_nums:
                patch_lines.append(num)
            # print("patch_lines: "+str(patch_lines))
            
            file_name_for_slice = patch_file_name.replace("/","$")
            backward_slicing_dir = slicing_dir + commit +"/"+status +"/"+file_name_for_slice+ "/backward"
            if os.path.exists(backward_slicing_dir):                 
                contents = helper.readFile(backward_slicing_dir )
                for line in contents:
                    try:
                        line_num = int(line.strip("\n"))
                    except ValueError:
                        print(commit+" failed on single_thread, backward slicing")
                        return counters
                    backward_slicing.append(line_num)
            # print("backward_slicing: "+str(backward_slicing))
            
            forward_slicing_dir = slicing_dir + commit +"/"+status +"/"+file_name_for_slice+"/forward"
            if os.path.exists(forward_slicing_dir):    
                contents = helper.readFile(forward_slicing_dir )
                for line in contents:
                    # print(line)
                    # print(forward_slicing_dir)
                    try:
                        line_num = int(line.strip("\n"))
                    except ValueError:
                        print(commit+" failed on single_thread, forward slicing")
                        return counters
                    forward_slicing.append(line_num)
            all_line_nums = list(backward_slicing) + list(forward_slicing) + list(patch_lines)
            all_line_nums.sort()
            out_commit_file = out_dir +commit+"/"
            helper.create_dir_if_not_exist(out_commit_file)
            out_file = out_commit_file + file_name_for_slice
            helper.delFileIfExists(out_file)
            
            # print("commit: "+commit+" file: "+patch_file_name+" len(file_lines): "+str(len(file_lines))+" all_line_nums: "+str(all_line_nums))
            # print("backward_slicing: "+str(backward_slicing)+" forward_slicing: "+str(forward_slicing)+" patch_lines: "+str(patch_lines))
            if all_line_nums[-1] -1>= len(file_lines):
                all_line_nums = patch_lines
                all_line_nums.sort()
                # print(all_line_nums)
                if all_line_nums[-1] -1>= len(file_lines):
                    continue
            file_handle = open(out_file, 'a+')
            for line_num in range(1,all_line_nums[-1]+1):
                if line_num - 1> len(file_lines) :
                    if commit not in wrong_slcings_commits:
                        wrong_slcings_commits.append(commit)
                    # print(f"{patch_file_name} one case line num >")
                    break
                if line_num in backward_slicing:
                    if file_lines[line_num-1] == "":
                        file_handle.write("\n")
                    else:
                        file_handle.write(backward_symbol+" "+file_lines[line_num-1]+"\n")
                elif line_num in patch_lines:
                    if file_lines[line_num-1] == "":
                        file_handle.write("\n")
                    else:
                        file_handle.write(patch_symbol+" "+file_lines[line_num-1]+"\n")
                elif line_num in forward_slicing:
                    if file_lines[line_num-1] == "":
                        file_handle.write("\n")
                    else:
                        file_handle.write(forward_symbol+" "+file_lines[line_num-1]+"\n")
                else:
                    file_handle.write("\n")
            file_handle.close()
    return counters

def gen_slicing_patch1(statuses, raw_data_dir,slicing_dir,symbols,commit_status_symbols,directions,backward_symbols,patch_symbols,forward_symbols):
    commits = []
    done_commits = []
    for i in range(len(statuses)):
        status = statuses[i]
        out_dir = raw_data_dir + status + "/"
        helper.create_dir_if_not_exist(out_dir)
    # print(out_dir)
    # for file in os.listdir(out_dir):
    #     if os.listdir(out_dir+file) != []:
    #         done_commits.append(file)

    print("done commits: "+str(len(done_commits)))
    for file in os.listdir(slicing_dir):
        if file not in done_commits:
            commits.append(file)
    
    
    print("unprocessed commits: "+str(len(commits)))

    with multiprocessing.Pool(60) as pool:
        count = 0
        counters = [0,0]
        for counter in pool.imap(single_thread,commits):
            count += 1
            if counter[0] == 1:
                counters[0] += 1
            if counter[1] == 1:
                counters[1] += 1
            if count % 100000 == 0:
                print(f"processed {count} commits ")
                print("counters: ", counters)
        print(counters)
    
def diff(file1, file2):
    cmd = "git diff --no-index "+file1.replace("$","\$")+" "+file2.replace("$","\$")
    res = helper.command(cmd)
    return res

def diff1(file1, file2):
    cmd = "git diff --no-index "+file1.replace("$","\$")+" "+file2.replace("$","\$")
    p = subprocess.run(cmd, shell=True, stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    res = p.stdout.decode("utf-8")
    return res

# def gen_slicing_diff(statuses, raw_data_dir,slicing_diff_dir):    
#     helper.create_dir_if_not_exist(slicing_diff_dir)
#     commits = []
#     for index in range(len(statuses)):
#         status = statuses[index]
#         out_dir = raw_data_dir + status + "/"
#         for commit in os.listdir(out_dir):
#             if commit not in commits:
#                 commits.append(commit)
    
#     print("commits: "+str(len(commits)))
#     for commit in tqdm.tqdm(commits):
#         commit_before_dir = raw_data_dir + statuses[0] + "/" + commit + "/"
#         commit_after_dir = raw_data_dir + statuses[1] + "/" + commit + "/"
#         commit_files = []
#         if os.path.exists(commit_before_dir):
#             for file in os.listdir(commit_before_dir):
#                 if file not in commit_files:
#                     commit_files.append(file)
#         if os.path.exists(commit_after_dir):
#             for file in os.listdir(commit_after_dir):
#                 if file not in commit_files:
#                     commit_files.append(file)
#         slicing_diff = []
#         for commit_file in commit_files:
#             commit_file_path1 = commit_before_dir + commit_file
#             commit_file_path2 = commit_after_dir + commit_file
#             if os.path.exists(commit_file_path1) and os.path.exists(commit_file_path2):
#                 res = diff(commit_file_path1, commit_file_path2)
#                 start = False
#                 for index in range(len(res)):
#                     line = res[index]
#                     line = line.strip("\n")
#                     if line.startswith("@@ -"):
#                         start = True
#                         continue
#                     if start:
#                         if line == " ":
#                             continue
#                         line = list(line)
#                         line = line[1:]
#                         line = "".join(line)
#                         slicing_diff.append(line)
#             if os.path.exists(commit_file_path1) and not os.path.exists(commit_file_path2):
#                 lines = helper.readFile(commit_file_path1)
#                 for line in lines:
#                     if line == "\n":
#                         continue
#                     slicing_diff.append(line.strip("\n"))
#             if not os.path.exists(commit_file_path1) and os.path.exists(commit_file_path2):
#                 lines = helper.readFile(commit_file_path2)
#                 for line in lines:
#                     if line == "\n":
#                         continue
#                     slicing_diff.append(line.strip("\n"))
#         out_file = slicing_diff_dir + commit + ""
#         helper.delFileIfExists(out_file)
#         for line in slicing_diff:
#             helper.dump(out_file, line.strip("\n"))
#         helper.dump(out_file, "\n")
        
def single_diff_thread(commit):
    commit_before_dir = raw_data_dir + statuses[0] + "/" + commit + "/"
    commit_after_dir = raw_data_dir + statuses[1] + "/" + commit + "/"
    commit_files = []
    if os.path.exists(commit_before_dir):
        for file in os.listdir(commit_before_dir):
            if file not in commit_files:
                commit_files.append(file)
    if os.path.exists(commit_after_dir):
        for file in os.listdir(commit_after_dir):
            if file not in commit_files:
                commit_files.append(file)
    slicing_diff = []
    for commit_file in commit_files:
        input_file = commit_file.replace("$","/")
        commit_file_path1 = commit_before_dir + commit_file
        commit_file_path2 = commit_after_dir + commit_file
        # if contain_path:
        #     slicing_diff.append(input_file)
        if os.path.exists(commit_file_path1) and os.path.exists(commit_file_path2):
            res = diff1(commit_file_path1, commit_file_path2)
            res = res.split("\n")
            # res = diff(commit_file_path1, commit_file_path2)
            start = False
            for index in range(len(res)):
                line = res[index]
                line = line.strip("\n")
                if line.startswith("@@ -"):
                    start = True
                    continue
                if start:
                    if line == " " or line == "":
                        continue
                    line = list(line)
                    line = line[1:]
                    if len(line) == 0:
                        continue
                    line = "".join(line)
                    slicing_diff.append(" "+line)
        if os.path.exists(commit_file_path1) and not os.path.exists(commit_file_path2):
            lines = helper.readFile(commit_file_path1)
            for line in lines:
                if line == "\n" or "@@ -" in line:
                    continue
                slicing_diff.append(" "+line.strip("\n"))
        if not os.path.exists(commit_file_path1) and os.path.exists(commit_file_path2):
            lines = helper.readFile(commit_file_path2)
            for line in lines:
                if line == "\n" or "@@ -" in line:
                    continue
                slicing_diff.append(" "+line.strip("\n"))
    out_file = slicing_diff_dir + commit + ""
    helper.delFileIfExists(out_file)
    for line in slicing_diff:
        helper.dump(out_file, line.strip("\n"))
    helper.dump(out_file, "\n")  
    return 1

def gen_slicing_diff1(statuses, raw_data_dir,slicing_diff_dir): 
    helper.create_dir_if_not_exist(slicing_diff_dir)
    commits = []
    for index in range(len(statuses)):
        status = statuses[index]
        out_dir = raw_data_dir + status + "/"
        #print(out_dir)
        for commit in os.listdir(out_dir):
            commits.append(commit)
    commits = [*set(commits)]
    #print("commits: "+str(len(commits)))

    with multiprocessing.Pool(60) as pool:
        count = 0
        for i in pool.imap(single_diff_thread,commits):
            count += 1
            if count % 100000 == 0:
                print(f"processed {count} commits diff")

def func_renaming_thread(commit):
    for status in statuses:
        commit_status_dir = raw_data_dir + status + "/" + commit + "/"
        new_commit_status_dir = new_raw_data_dir + status + "/" + commit + "/"
        helper.create_dir_if_not_exist(new_commit_status_dir)
        if os.path.exists(commit_status_dir):
            for patched_file in os.listdir(commit_status_dir):
                new_patched_file = new_commit_status_dir + patched_file
                helper.del_file_if_exists(new_patched_file)
                lines = helper.readFile(commit_status_dir + patched_file)
                for line in lines:
                    start = False
                    if "etype = inode_bmap(inode, first_block" in line:
                        start = True
                    if line == "\n":
                        helper.dump(new_patched_file, line)
                        continue
                    for old_func, new_func in func_renaming_dict.items():
                        if " "+old_func+"(" in line:
                            line = line.replace(" "+old_func+"(", " "+new_func+"(")
                            # if start:
                            #     print("old_func: "+old_func)
                            #     print("before replacing, line: "+line)
                            #     line = line.replace(" "+old_func+"(", " "+new_func+"(")
                            #     print("after replacing, line: "+line)
                    helper.dump(new_patched_file, line)
                    


def func_count_thread(commit,slicing_diff_dir):
    num, diffnum = 0, 0
    patch_symbols = ["\u240d","\u241d"]
    result = []
    
    lines = helper.readFile(slicing_diff_dir + commit)
    for line in lines:
        for old_func, new_func in func_renaming_dict.items():
            if " "+old_func+"(" in line:
                num += 1
                result.append(old_func)
    return num, len(list(set(result))),result

    # for status in statuses:
    #     commit_status_dir = raw_data_dir + status + "/" + commit + "/"
    #     if os.path.exists(commit_status_dir):
    #         for patched_file in os.listdir(commit_status_dir):
    #             lines = helper.readFile(commit_status_dir + patched_file)
    #             for line in lines:
    #                 if line.startswith(patch_symbols[0])==False and line.startswith(patch_symbols[1])==False:
    #                     continue
    #                 start = False
    #                 if "etype = inode_bmap(inode, first_block" in line:
    #                     start = True
    #                 if line == "\n":
    #                     # helper.dump(new_patched_file, line)
    #                     continue
    #                 for old_func, new_func in func_renaming_dict.items():
    #                     if " "+old_func+"(" in line:
    #                         num += 1
    #                         result.append(old_func)
                            
    # return num, len(list(set(result)))

def func_count(raw_data_dir,func_renaming_dict): 
    global statuses
    statuses = ["before_patch","after_patch"]
    commits = []
  
    slicing_diff_dir = "xxx"
    for file in os.listdir(slicing_diff_dir):
        commits.append(file)
    print("commits: "+str(len(commits)))
    
    global chosen
    chosen = []
    src_commits_dir = "xxx"
    for line in helper.readFile(src_commits_dir+"cveeval_commits"):
        chosen.append(line.strip("\n"))
    print("chosen: "+str(len(chosen)))
    src_file =  "xx"
    func_renaming_dict = ast.literal_eval(helper.readFile(src_file)[0])
    del func_renaming_dict[""]
    start_time = time.time()
    result = []
    result1 = []
    eval_funcs = []
    with multiprocessing.Pool(60) as pool:
        count = 0
        # for i in pool.imap(func_renaming_thread,commits):
        for i in pool.imap(func_count_thread,chosen):
            num, diffnum, funcs = i[0], i[1],i[2]
            result.append(num)
            result1.append(diffnum)
            eval_funcs.extend(funcs)
            # if count % 100 == 0:
            #     print(f"processed {count} commits diff")
            #     print("duration since start: {}".format(time.time()-start_time))
    print("result: "+str(sum(result)/len(result)))
    print("result1: "+str(sum(result1)/len(result1)))
    print("eval_funcs: "+str(len(list(set(eval_funcs)))))
    count = len(list(set(eval_funcs)))
    print("eval_funcs: "+str(count/len(chosen)))
    
    
    chosen = []
    src_commits_dir = "xxx"
    for line in helper.readFile(src_commits_dir+"train_commits"):
        chosen.append(line.strip("\n"))
    print("chosen: "+str(len(chosen)))
    result = []
    result1 = []
    train_funcs = []
    with multiprocessing.Pool(60) as pool:
        count = 0
        # for i in pool.imap(func_renaming_thread,commits):
        for i in pool.imap(func_count_thread,chosen):
            num, diffnum, funcs = i[0], i[1],i[2]
            result.append(num)
            result1.append(diffnum)
            train_funcs.extend(funcs)
            # if count % 100 == 0:
            #     print(f"processed {count} commits diff")
            #     print("duration since start: {}".format(time.time()-start_time))
    print("result: "+str(sum(result)/len(result)))
    print("result1: "+str(sum(result1)/len(result1)))
    print("train_funcs: "+str(len(list(set(train_funcs)))))
    count = len(list(set(train_funcs)))
    print("train_funcs: "+str(count/len(chosen)))
    
    count = 0
    eval_funcs = list(set(eval_funcs))
    train_funcs = list(set(train_funcs))
    for func in eval_funcs:
        if func not in train_funcs:
            count += 1
    print("funcs for eval: "+str(len(eval_funcs)))
    print("funcs in eval not in train: "+str(count))
    for func in random.sample(eval_funcs,250):
        helper.dump("./chosen_evaul_funcs",func+"\n")
    for func in random.sample(train_funcs,250):
        helper.dump("./chosen_train_funcs",func+"\n")

def func_renaming(raw_data_dir,new_raw_data_dir,func_renaming_dict,src_file): 
    global statuses
    statuses = ["before_patch","after_patch"]
    helper.create_dir_if_not_exist(new_raw_data_dir)
    commits = []
    for index in range(len(statuses)):
        status = statuses[index]
        out_dir = raw_data_dir + status + "/"
        print(out_dir)
        for commit in os.listdir(out_dir):
            commits.append(commit)
    commits = [*set(commits)]
    print("commits: "+str(len(commits)))
    func_renaming_dict = ast.literal_eval(helper.readFile(src_file)[0])
    del func_renaming_dict[""]
    start_time = time.time()
    with multiprocessing.Pool(60) as pool:
        count = 0
        for i in pool.imap(func_renaming_thread,commits):
            count += 1
            # if count % 100 == 0:
            #     print(f"processed {count} commits diff")
            #     print("duration since start: {}".format(time.time()-start_time))

 
def sets_split1(set_dir,slicing_diff_dir,splits,goal,with_title,title_from_file=False,cveeval_type="",given_slicing_diff_dir=""):
    commits = []
    if goal == "pretrain":    
        for file in os.listdir(slicing_diff_dir):
            commits.append(file)
        random.shuffle(commits)
        length = len(commits)
        print(length)
        commits = [*set(commits)]
        print(len(commits))
        train = commits[:int(length*0.9)]
        valid = commits[int(length*0.9):]
        set_commits = [train,valid]
    elif goal == "finetune":
        src_file = PROJECT_DIR + "/raw_data/all_data" #this file is after removing dirty data, such as OOB in bluetooth
        lines = helper.readFile(src_file)
        all_data = dict()
        for line in lines:
            commit = line.split(": ")[0]
            label = line.split(": ")[1].strip("\n")
            all_data[commit] = str(int(label)-1)
        for file in os.listdir(slicing_diff_dir):
            if file  in all_data:#this is to confirm that the commit does not contain cve commits which are finally used for evaluation
                commits.append(file)
       
        random.shuffle(commits)
        length = len(commits)
        train = commits[:int(length*0.8)]
        valid = commits[int(length*0.8):int(length*0.9)]
        test = commits[int(length*0.9):]
        
        set_commits = [train,valid,test]
    elif goal == "given":
        eval_commits = [x for x in os.listdir(given_slicing_diff_dir)]
        # labels = ["6"] * len(eval_commits)
        # all_data = given_lables
        all_data = {commit:"6" for commit in eval_commits}
        set_commits = [eval_commits]
   
        
    print("len(set_commits): {}".format(len(set_commits)))
    helper.create_dir_if_not_exist(set_dir)
    for index in range(len(set_commits)):
        commits = set_commits[index]
        split = splits[index]
        set_file = set_dir + split + "_commits"
        set_input = set_dir + split + ".input"
        if goal != "pretrain":
            label_input = set_dir + split + ".label"
            helper.delFileIfExists(label_input)
        helper.delFileIfExists(set_file)
        helper.delFileIfExists(set_input)
        args = []
        print("for {} set, {} commits".format(split,len(commits)))
        if title_from_file:
            commit_tile_pairs = helper.readFile("xxx")
            commit_tile_pairs = ast.literal_eval(commit_tile_pairs[0])
        else:
            commit_tile_pairs = {}
        for commit in commits:
            if goal != "pretrain":
                if commit not in all_data.keys():
                    print(commit+" not in ")
                    continue
            args.append((commit,slicing_diff_dir,with_title,title_from_file,commit_tile_pairs))
        with multiprocessing.Pool(60) as pool:
            temp = dict()
            for pair in pool.imap(set_split_thread,args):
                commit, inputs = pair
                if commit != -1 and inputs != -1:
                    temp[commit] = inputs
            for commit, inputs in temp.items():
                helper.dump(set_file, commit+"\n")
                helper.dump(set_input, inputs)
                if goal != "pretrain":
                    if "," in all_data[commit]:
                        label = all_data[commit].split(",")[0]
                    else:
                        label = all_data[commit]
                    helper.dump(label_input, label+"\n")

    print("\n\n\n SETS SPLIT DONE \n\n\n")   



def set_split_thread(args):
    is_pretrain = True
    commit,slicing_diff_dir,with_title, title_from_file, commit_tile_pairs = args
    src_file = slicing_diff_dir + commit
    if os.path.exists(src_file) == False:
        print(f"file {src_file} not exist")
        return -1,-1
   
    inputs = ""
    if with_title:
        if title_from_file:
            if commit in commit_tile_pairs.keys():
                title = commit_tile_pairs[commit]
            else:
                title = helper.get_Title(repo_dir,commit)
        else:
            title = helper.get_Title(repo_dir,commit)
        inputs += title.strip("\n")
    lines = helper.readFile(src_file)  
    if len(lines) != 0:
        if lines[0] != "\n":
            lines[0] = lines[0][1:]
    for line in lines:
        inputs += line.strip("\n")
    inputs += "\n"
    res = (commit, inputs)
    
    return res

def train_spm(set_dir,sets,sentencepiece_dir,SPMVOCAB,DICT_FILE):
    # patches = os.listdir(raw_patches)
    # FILES = [raw_patches + file for file in patches]
    FILES = [set_dir + set + ".input" for set in sets]
    spm.SentencePieceTrainer.train(
        '--input={} --vocab_size=50000 --model_prefix=sentencepiece.bpe '
        '--character_coverage=1.0 --model_type=bpe --max_sentence_length=16192 --num_threads=40'.format(','.join(FILES))
    )
    
    if helper.run_cmd_return_status("mv sentencepiece.bpe.model "+sentencepiece_dir) != 0:
        print("mv sentencepiece.bpe.model failed")
        return
    if helper.run_cmd_return_status("mv sentencepiece.bpe.vocab "+sentencepiece_dir) != 0:
        print("mv sentencepiece.bpe.vocab failed")
        return
    dict_cmd = "cut -f1 {} | tail -n +4 | sed \"s/$/ 100/g\" >{}".format(SPMVOCAB,DICT_FILE)
    if helper.run_cmd_return_status(dict_cmd) != 0:
        print("dict generation failed")
        return
    print("\n\n\n DICT GENERATION DONE! \n\n\n")        

def spm_process(sets, set_dir, spm_dir, SPMMODEL,sample_limit):
    #print(sample_limit)
    if sample_limit == "long":
        max_len = 2040
    if sample_limit == "medium":
        max_len = 1020
    if sample_limit == "short":
        max_len = 510
    for split in sets:
        src_file = set_dir + split + ".input"
        out_file = spm_dir+ split+".spm"
        lines = helper.readFile(src_file)
        helper.delFileIfExists(out_file)
        cmd = "python "+ PROJECT_DIR +"/codes/encode.py --model-file "+SPMMODEL+" --inputs "+src_file +" --outputs  "+out_file+" --max_len "+str(max_len)+" --workers 60"
        #print(cmd)
        res,out = helper.run_cmd_return_status1(cmd)
        if res != 0:
            print("spm preprocess failed")
            return
        else:
            print(out)
    print("\n\n\n SPM PREPROCESS DONE! \n\n\n")
    
def binarize(data_dir,spm_dir,splits,DICT_FILE, with_label):
    helper.delFileIfExists(data_dir+ "binarize/label/dict.txt")
    prefs = ["trainpref","testpref","validpref"]
    preprocess_cmd = f"fairseq-preprocess --only-source  --destdir "+data_dir + "binarize/label/ --workers 60"
    for index in range(len(splits)):
        split = splits[index]
        pref = split+"pref"
        src_file = spm_dir+ split+".spm"
        if with_label:
            out_file = data_dir + "binarize/input0/"
        else:
            out_file = data_dir + "binarize/"
        cmd = f"fairseq-preprocess --only-source --{pref} {src_file}  --destdir {out_file} --srcdict {DICT_FILE} --workers 60"
        print(cmd)
        if helper.run_cmd_return_status(cmd) != 0:
            print("binarize "+split+" failed")
            return
        if with_label:
            src_file = data_dir + "train_valid_test/"+ split+".label"
            preprocess_cmd += f" --{pref} {src_file} "
    if with_label:
        print(preprocess_cmd)
        if helper.run_cmd_return_status(preprocess_cmd) != 0:
            print("binarize label failed")
            return
    print("\n\n\n BINARIZE DONE! \n\n\n")

def process(args,pretrain_save_dir,out_dir,data_dir,slicing_dir):    
    print(data_dir)
    helper.create_dir_if_not_exist(data_dir)   
    global contain_path
    contain_path = False
    if args.filepath:
        contain_path = True
   
    global statuses
    global raw_data_dir
    global symbols
    global commit_status_symbols
    global directions
    global backward_symbols
    global patch_symbols 
    global forward_symbols
    global slicing_diff_dir
  
    title_from_file = False
  
    slicing_diff_dir =  data_dir + "slicing_diff/"#" 
    split_dir = data_dir + "train_valid_test/"
    spm_preprocess = data_dir + "spm_preprocess/"
    binarize_dir = data_dir + "binarize/"
    if args.goal == "pretrain":
        sentencepiece_dir = data_dir + "sentencepiece/"
    else:
        sentencepiece_dir = HOME_DIR + "/data/"+pretrain_save_dir + "/sentencepiece/"
 
    sentencepiece_dir = "xx/xx"
    SPMMODEL = sentencepiece_dir + "sentencepiece.bpe.model"
    SPMVOCAB = sentencepiece_dir + "sentencepiece.bpe.vocab"
    DICT_FILE = sentencepiece_dir + "dict.txt"
    print(f"SPMMODEL: {SPMMODEL}")
    # exit()
    helper.create_dir_if_not_exist(raw_data_dir)
    helper.create_dir_if_not_exist(split_dir)
    helper.create_dir_if_not_exist(spm_preprocess)
    helper.create_dir_if_not_exist(binarize_dir)
    helper.create_dir_if_not_exist(sentencepiece_dir)
    
    statuses = ["before_patch","after_patch"]
    commit_status_symbols = ["^",""]
    directions = ["forward","backward"]
    symbols = ["-","+"]
    special_tokens = True
    if special_tokens:
        backward_symbols = ["\u240c","\u241c"]
        patch_symbols = ["\u240d","\u241d"]
        forward_symbols = ["\u240e","\u241e"]
    else:
        backward_symbols = ["",""]
        patch_symbols = ["-","+"]
        forward_symbols = ["",""]
    
    if args.goal == "pretrain":
        splits = ["train","valid"]
    elif args.goal == "finetune":
        splits = ["train","valid","test" ]
 
         
    sets_split1(split_dir,slicing_diff_dir,splits, args.goal ,args.title == "with",title_from_file)
    
    if args.goal == "pretrain":
        train_spm(split_dir, splits,sentencepiece_dir,SPMVOCAB,DICT_FILE)
    
    spm_process(splits, split_dir, spm_preprocess, SPMMODEL, args.length) 
   
    binarize(data_dir,spm_preprocess,splits,DICT_FILE, args.goal != "pretrain")
 
def given_slice1(commits,name,repo_dir1):
    from slice import slice1_thread
    global repo_dir
    repo_dir = repo_dir1
    global depends_out
    global depends_edges
    depends_out = HOME_DIR+"/slices/"+name+"/depends_edges/"
    helper.create_dir_if_not_exist(depends_out)
    depends_edges = depends_out

    global ddgs_out
    ddgs_out = HOME_DIR+"/slices/"+name+"/slices1/"
    helper.create_dir_if_not_exist(ddgs_out)
    global distance
    distance = 0

    # commits = os.listdir(depends_out)
    print("Generate ddg for "+name+" commits starts")
    count = 0
    invalid = []
    with multiprocessing.Pool(30) as pool:
        for i in pool.imap(slice1_thread,commits):
            count += 1
            print(count)
            if i[0] == -1:
                invalid.append(i[1])
            if count % 100 == 0:
                print("processed {} commits".format(count))
                
    print("invalid commits: "+str(invalid))
    
def given_slice2(commits,name,repo_dir1):
    from slice import joern_slice
    global repo_dir
    repo_dir = repo_dir1
    global depends_out
    global depends_edges
    global done_out
    depends_out = HOME_DIR+"/slices/"+name+"/depends_edges_labels/"
    depends_edges = depends_out
     

    with multiprocessing.Pool(20) as pool:
        counter = 0
        for i in pool.imap(joern_slice,commits):
            counter += 1
            if counter % 100 == 0:
                print("counter: {}".format(counter))

def func_renaming1(name,src_dir):
    import re
    global func_renaming_dict
    src_file = HOME_DIR +"/data/function_renames07_parsed"
    func_renaming_dict = ast.literal_eval(helper.readFile(src_file)[0])
    del func_renaming_dict[""]
    src_file = src_dir+name+".input"
    out_file = src_dir+name+".input.renamed"
    helper.delFileIfExists(out_file)
    lines = helper.readFile(src_file)
    # Step 2: Order patterns from longest to shortest
    patterns = sorted(func_renaming_dict.keys(), key=len, reverse=True)
    # Step 3: Create the regex pattern with word boundaries and capturing group
    pattern = r'\b(' + '|'.join(re.escape(p) for p in patterns) + r')\b'
    # Step 4: Define the replacement function
    def replace_match(match):
        # Get the matched text
        matched_text = match.group(0)
        # Return the corresponding replacement
        return func_renaming_dict[matched_text]
    for line in lines:
        result = re.sub(pattern, replace_match, line)
        helper.dump(out_file,result)

def get_lines(src_file):
        return [line.strip("\n") for line in helper.readFile(src_file)]
    
def pair_dump(commits_file, inputs_file):
    commits = get_lines(commits_file)
    inputs = get_lines(inputs_file)
    all_con = dict(zip(commits,inputs))
    # for commit, input1 in all_con.items():
    #     helper.dump(out_file1,commit+"\n")
    #     helper.dump(out_file2,input1+"\n")
    return all_con
     
 
def split_summary_slicingdiff(input1):
    symbols = ["\u240c","\u241c","\u240d","\u241d","\u240e","\u241e"]
    raw = input1

    for symbol in symbols:
        if symbol in input1:
            input1 = input1.split(symbol)[0]
    return input1,raw.replace(input1,"")

 

     
    
         
    
def given_preprocess_ffempeg_openssl(slicing_dir,proj):
    global statuses
    global raw_data_dir
    global symbols
    global commit_status_symbols
    global directions
    global backward_symbols
    global patch_symbols 
    global forward_symbols
    global slicing_diff_dir
    global contain_path
    contain_path = False
    
    global repo_dir
    repo_dir = "repos/openssl/"
    
 
    
    data_dir = HOME_DIR + "/data/"+proj+"_bti/"
    helper.create_dir_if_not_exist(data_dir)
    split_dir = data_dir + "train_valid_test/"
    helper.create_dir_if_not_exist(split_dir)
    spm_preprocess = data_dir + "spm_preprocess/"
    helper.create_dir_if_not_exist(spm_preprocess)
    slicing_diff_dir = data_dir + "slicing_diff/"
    helper.create_dir_if_not_exist(slicing_diff_dir)
    
    sentencepiece_dir = "xxx/xxx"
    DICT_FILE = sentencepiece_dir + "dict.txt"
    SPMMODEL = sentencepiece_dir + "sentencepiece.bpe.model"

    backward_symbols = ["\u240c","\u241c"]
    patch_symbols = ["\u240d","\u241d"]
    forward_symbols = ["\u240e","\u241e"]
    statuses = ["before_patch","after_patch"]
    commit_status_symbols = ["^",""]
    directions = ["forward","backward"]
    symbols = ["-","+"]
    raw_data_dir = data_dir + "raw_data/"
    gen_slicing_patch1(statuses, raw_data_dir,slicing_dir,symbols,commit_status_symbols,directions,backward_symbols,patch_symbols,forward_symbols)
    gen_slicing_diff1(statuses, raw_data_dir,slicing_diff_dir)
    
 
    splits = ["eval"]
    sets_split1(split_dir,slicing_diff_dir,splits, "btitreevul" ,True,"",given_slicing_diff_dir=slicing_diff_dir)
    
    spm_process(splits, split_dir, spm_preprocess, SPMMODEL, "medium")

 
    
def context_given(data_dir,src_dir,src_summary_file,commit_truth):
    global statuses
    global raw_data_dir
    global slicing_dir
    global symbols
    global commit_status_symbols
    global directions
    global backward_symbols
    global patch_symbols 
    global forward_symbols
    global slicing_diff_dir
    global contain_path
    contain_path = False
    commit_summaries= ast.literal_eval(helper.readFile(src_summary_file)[0])
    helper.create_dir_if_not_exist(data_dir)
    split_dir = data_dir + "train_valid_test/"
    helper.create_dir_if_not_exist(split_dir)
    spm_preprocess = data_dir + "spm_preprocess/"
    helper.create_dir_if_not_exist(spm_preprocess)
    out_file = split_dir + "eval.input"
    out_file1 = split_dir + "eval.label"
    out_file2 = split_dir + "eval_commits"
    helper.del_file_if_exists(out_file)
    helper.del_file_if_exists(out_file1)
    for commit in commit_truth:
        title = helper.get_Title(linux_dir, commit)
        diff = helper.get_commitContents(linux_dir,commit,True)
        diff = [line.strip("\n") for line in diff ]
        diff = "".join(diff)
        if commit in commit_summaries:
            helper.dump(out_file,"$"+commit_summaries[commit]+"$"+diff+"\n")
        else:
            helper.dump(out_file,"$"+title.strip("\n")+"$"+diff+"\n")
        helper.dump(out_file1,commit_truth[commit]+"\n")
        helper.dump(out_file2,commit+"\n")
    sentencepiece_dir = "xxx/xxx"
    DICT_FILE = sentencepiece_dir + "dict.txt"
    SPMMODEL = sentencepiece_dir + "sentencepiece.bpe.model"
    splits = ["eval"]
    spm_process(splits, split_dir, spm_preprocess, SPMMODEL, "medium")

 
            

def get_context(commit,out_file):
    contents = []
    cmd = "cd "+linux_dir+";git show -m "+commit
    result = helper.command(cmd)
    contents_start = False
    for line in result:
        if line.startswith("diff"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") or line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("@@ "):
                pieces = line.split("@@ ")
                if len(pieces) == 3:
                    contents.append(line.split("@@ ")[-1])
            else:
                contents.append(line)
    helper.delFileIfExists(out_file)
    title = helper.get_Title(linux_dir,commit)
    helper.dump(out_file, title.strip("\n")+"\u241d\n")
    for line in contents:
        helper.dump(out_file, line)
#name: the name of evaluation dataset
#evaluation_commits: the commits used for evaluation
#src_summary_file: the file containing the summary of each commit, can be generated by the function get_summaries() of llm_query.py
#data_dir: the directory to store the preprocessed data


def build_eval_data_for_random_given(name,evaluation_commits,src_summary_file,data_dir):    
    global repo_dir
    repo_dir = linux_dir
    global statuses
    global raw_data_dir
    global slicing_dir
    global symbols
    global commit_status_symbols
    global directions
    global backward_symbols
    global patch_symbols 
    global forward_symbols
    global slicing_diff_dir
    backward_symbols = ["\u240c","\u241c"]
    patch_symbols = ["\u240d","\u241d"]
    forward_symbols = ["\u240e","\u241e"]
    statuses = ["before_patch","after_patch"]
    commit_status_symbols = ["^",""]
    directions = ["forward","backward"]
    symbols = ["-","+"]
    raw_data_dir = data_dir + "/raw_data/"
    helper.create_dir_if_not_exist(raw_data_dir)
    spm_preprocess = data_dir + "/spm_preprocess/"
    helper.create_dir_if_not_exist(spm_preprocess)
    split_dir = data_dir+"/train_valid_test/"
    helper.create_dir_if_not_exist(split_dir)
    slicing_diff_dir = data_dir + "/slicing_diff/"
    commits_labels = {commit:"0" for commit in evaluation_commits}
    helper.create_dir_if_not_exist(slicing_diff_dir)
    
    sentencepiece_dir =HOME_DIR+"/codes/sentencepiece/"
    DICT_FILE = sentencepiece_dir + "dict.txt"
    SPMMODEL = sentencepiece_dir + "sentencepiece.bpe.model"   
    from slice import given_slice
    given_slice(evaluation_commits,name,linux_dir)
    slicing_dir = HOME_DIR+"/slices/"+name+"/slices_onlydata" 
    # given_slice1(evaluation_commits,name,linux_dir)
    # given_slice2(evaluation_commits,name,linux_dir)
    # slicing_dir = HOME_DIR+"/slices/"+name+"/depends_edges_labels"
    splits=["eval"]
    def get_slicingdiff():
        gen_slicing_patch1(statuses, raw_data_dir,slicing_dir,symbols,commit_status_symbols,directions,backward_symbols,patch_symbols,forward_symbols)
        gen_slicing_diff1(statuses, raw_data_dir,slicing_diff_dir)
        with_title=False
        
        sets_split1(split_dir,slicing_diff_dir,splits, "given" ,with_title,"",given_slicing_diff_dir=slicing_diff_dir)#
 
    def combine_slicingdiff_summary():
        current_dir = data_dir+"/train_valid_test/"
        commits_inputs = pair_dump(current_dir+"/eval_commits",current_dir+"/eval.input.renamed")
        # commits_inputs = pair_dump(current_dir+"/eal_onlyslicingdiff_commits",current_dir+"/eval_onlyslicingdiff.input")
        commit_summaries = ast.literal_eval(helper.readFile(src_summary_file)[0])
        for commit in commits_inputs.keys():
            summary = commit_summaries[commit]
      
            helper.dump(current_dir+"eval.input",summary+commits_inputs[commit]+"\n")
            helper.dump(current_dir+"eval_commits", commit+"\n")
            helper.dump(current_dir+"eval.label", commits_labels[commit]+"\n")
        splits=["eval"]
        spm_process(splits, current_dir, data_dir+"/spm_preprocess/", SPMMODEL, "medium")
  
    get_slicingdiff()
    # move()
    func_renaming1("eval",data_dir+"/train_valid_test/")
    combine_slicingdiff_summary()