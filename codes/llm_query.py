import sys
from pathlib import Path
sys.path.append("../../")
import helper
import openai
import os
import random
import argparse
import ast
import tiktoken
import json
import time
import re
import requests

cwd = os.path.dirname(os.path.realpath(__file__))
linux_dir = str(Path(cwd).parent.absolute()) + "/repos/linux"
HOME_DIR = str(Path(cwd).parent.absolute())
PROJECT_DIR = str(Path(cwd).parent.parent.absolute())
 
# Get OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")
llama_key="xxx"  
TEMPERATURE =1.0
 
 

def gpt4(prompt_messages, passed_tempature):
    response = openai.ChatCompletion.create(model="gpt-4", messages=prompt_messages, temperature=passed_tempature, max_tokens=512)
    # print("response: ", response)
    # exit()
    return response["choices"][0]["message"]["content"]

def gpt411(prompt_messages, passed_tempature):
    response = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=prompt_messages, temperature=passed_tempature, max_tokens=512)
    # print("response: ", response)
    # exit()
    return response["choices"][0]["message"]["content"]

def get_summary(commit):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": generate_prompt(commit)}], temperature=0, max_tokens=512)
    return response["choices"][0]["message"]["content"]

def gpt411(prompt_messages, passed_tempature):
    response = openai.ChatCompletion.create(model="gpt-4-1106-preview", messages=prompt_messages, temperature=passed_tempature, max_tokens=512)
    # print("response: ", response)
    # exit()
    return response["choices"][0]["message"]["content"]

def gpt_model(model1,prompt_messages, passed_tempature):
    response = openai.ChatCompletion.create(model=model1, messages=prompt_messages, temperature=passed_tempature, max_tokens=512)
    # print("response: ", response)
    # exit()
    return response["choices"][0]["message"]["content"]
def gpt_model_o1(model1,prompt_messages):
    response = openai.ChatCompletion.create(model=model1, messages=prompt_messages,  max_completion_tokens=24426)
    
    return response["choices"][0]["message"]["content"]

def num_tokens_from_prompt(prompt, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_prompt(prompt, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        # print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_prompt(prompt, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_prompt() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    num_tokens += tokens_per_message
    num_tokens += len(encoding.encode(prompt))
      
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def get_message(commit):
    cmd="cd "+linux_dir+";git log --format=%B -n 1 "+commit+" --pretty=fuller"
    result = helper.command(cmd)
    temp = result[6:]
    res = ""
    for line in temp:
        if line.startswith("    Signed-off-by") == True or line.startswith("    Reported-by")==True or line.startswith("    Fixes")==True  \
        or line.startswith("    Tested-by") == True or line.startswith("Cc:") == True:
            break
        res += line.strip("\n")
    return res

def get_patch(commit):
    cmd = "cd "+linux_dir+";git show "+commit+" --pretty=fuller"
    result = helper.command(cmd)
    result = result[6:]
    patch = ""
    for line in result:
        patch += line.replace("\t", "")
    return patch
    
    
def generate_prompt(commit):
    message = get_message(commit)
    
    prompt = "I want to you act as a Linux security patch expert. The commit title and commit message of a Linux security patch is usually too long,\\\
            Please summarize it into one sentence, focusing on the root cause the bug and how this bug is fixed.\""+message+"\"."
    return prompt
 
 
def get_commitContents(commit):
    diff = ""
    contents = []
    cmd = "cd "+linux_dir+";git show -m "+commit
    result = helper.command(cmd)
    contents_start = False
    for line in result:
        if line.startswith("diff"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") :
                continue
            diff+=line.replace("\t","")
            contents.append(line.replace("\t",""))
    return diff

def get_commitContents1(commit,not_symbol):
    diff = ""
    contents = []
    cmd = "cd "+linux_dir+";git show -m "+commit
    result = helper.command(cmd)
    contents_start = False
    for line in result:
        if line.startswith("diff --git"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") :
                continue
            
            if line.startswith(not_symbol):
                continue
            diff+=line.replace("\t","")
            contents.append(line.replace("\t",""))
    return diff

def get_commitContents2(commit,not_symbol):
    diff = ""
    contents = []
    cmd = "cd "+linux_dir+";git show -m "+commit
    result = helper.command(cmd)
    contents_start = False
    current_file = ""
    file_lines = dict()
    for line in result:
        if line.startswith("diff --git"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") or line.startswith("+++ ") or line.startswith("--- ") :
                if line.startswith("+++ ") or line.startswith("--- "):
                    current_file = line[4:].strip("\n")
                continue
            if line.startswith(not_symbol):
                continue
            if current_file not in file_lines.keys():
                file_lines[current_file] = ""
            file_lines[current_file]+=line.replace("\t","")
            diff+=line.replace("\t","")
            contents.append(line.replace("\t",""))
    return file_lines
 
    
def bug_type_by_slicing():
    raw_data_dir = HOME_DIR+"xxx"
    statues = ["before_patch", "after_patch"]  #first give you a summary of commit title and message, and then
    status_pair = {"before_patch":"Removed codes", "after_patch":"Added codes"}
    correct = 0
    passed_to_gpt = 0
    
    src_prompt = HOME_DIR+"/prompts/slicing_prompt1"
    lines = helper.readFile(src_prompt)
    fix_prompt = "".join(lines)
    out_file = HOME_DIR + "xxx" 
      
    todos=[line.split(" ")[0] for line in helper.readFile(HOME_DIR +"/xxx")]
    result={}
    for commit in todos:
       
        print(commit)
        title_message = helper.get_commitMessage(linux_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
        prompt = ""
        prompt += fix_prompt
        prompt += fix_prompt + "\nCommit title: "+title+"\n"
        prompt += "Commit message: "+message+""

        for status in statues:
            src_dir = raw_data_dir + status + "/" + commit
            if os.path.exists(src_dir):
                for file in os.listdir(src_dir):
                    src_file = src_dir + "/" + file
                    patched_file = file.replace("$","/")
                    lines = helper.readFile(src_file)
                    slicing_diff = ""
                    for line in lines:
                        if line == "\n":
                            continue
                        slicing_diff += line.replace("␍","-").replace("␝","+")
                    prompt += status_pair[status]+" and its code context for "+patched_file+":\n"+slicing_diff+"\n "
            else:
                prompt += "There is no "+status_pair[status]+" code snipet.\n"
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        if prompt_len+512 > 8192:
            result[commit] = "TOO LONG(>4096)"
        else:
            passed_to_gpt += 1
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            res = gpt4(message,0.7)
            #print(res)
            result[commit] = res
            if "Bug type: " in res:
                bug_type = res.split("Bug type: ")[1].split("\n")[0]
            else:
                bug_type = "NOT ANY"
            print("bug type: ",bug_type)#Based on the commit title, message, and code changes, the bug type is use-after-free.
     
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        #print("passed_to_gpt/total: "+str(passed_to_gpt)+"/"+str(len(todos)))
        
def bug_type_by_diff(commits,src_prompt):
    lines = helper.readFile(src_prompt)
    fix_prompt = "".join(lines)

    correct = 0
    passed_to_gpt = 0
    
    out_file = HOME_DIR + "/results/gpt/diff_3lines_gpt4_TEM"+str(TEMPERATURE).replace(".", "")
    
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
        passed_to_gpt += len(result.keys())
        for commit, type in result.items():
            if str(type2int(type)) ==commits[commit]:
                correct += 1
            # else:
            #     print(commit)
    else:
        result = dict()
    lines = helper.readFile(HOME_DIR +"/data/cves/cveeval_commits")
    cveeval_commits =[x.strip("\n") for x in lines]
    total = len(cveeval_commits)
    for commit in cveeval_commits:
        if commit in result.keys():
            continue
        prompt = ""
        prompt += fix_prompt
        diff_3lines = helper.get_diff(linux_dir,commit)
        diff_3lines = [x.strip("\n").strip() for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        if prompt_len+512 > 8192:
            result[commit] = "TOO LONG(>4096)"
        else:
            passed_to_gpt += 1
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            res = gpt4(message,0.7)
            #print(res)
            result[commit] = res
            if "Bug type: " in res:
                bug_type = res.split("Bug type: ")[1].split("\n")[0]
            else:
                bug_type = "NOT ANY"
            print("bug type: ",bug_type)
            if str(type2int(bug_type)) == commits[commit]:
                correct += 1
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        #print("correct/passed_to_gpt/total: "+str(correct)+"/"+str(passed_to_gpt)+"/"+str(total))

def type2int(bug_type):
    bug_type = bug_type.replace("\"", "").replace(".","").lower()
 
    oob_keywords= ["out of bound","out-of-bound","oob","buffer overflow","stack overflow"]
    uab_keywords = ["use-after-free","uaf","use after free","invalid free","double free"]
    for keyword in oob_keywords:
        if keyword in bug_type:
            return 0
    for keyword in uab_keywords:
        if keyword in bug_type:
            return 1
    return 2     

def prompt2gpt(prompt,commit,types):
    if num_tokens_from_prompt(prompt)+512 > 4096:
        return "TOO LONG(>4096)", 
    else:
        passed_to_gpt += 1
        res = gpt4(prompt)
        print("result from gpt-3.5 is ",res)
        if str(type2int(res)) == types:
            return 1
        else:
            return 0
        result[commit] = res
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
 
def bug_type_by_whole_patch(out_file,commits):
     
    
    correct = 0
    passed_to_gpt = 0 
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
        passed_to_gpt += len(result.keys())
        
    else:
        result = dict()
    
    print(passed_to_gpt)
    fix_prompt = "I want you to act as a Linux kernel security patch expert, who is great at analyzing Linux security patches. A linux kernel patch is composed of commit title, message and diff.\\\
        Commit title and message may imply the root cause of the bugs fixed by the patch and how this patch fixes this bug. Commit diff is the differences for the patched files between the time \\\
            when you are unpatched and the time when they are patched. I need your help to tell me the bug type of a give patch. I will give you commit title, message, diff and three lines around diff as context. I hope that you can first look at commit title and commit message, \\\
                and then commit diff and its context; you can try to understand the root cause of the bugs fixed by the patch. There are three possible bug types: use-after-free, memory out-of-bounds, and non-uaf-oob(including memory leak, use before initilization, null poiner derference and other bug types)\\\
                You can choose one of bug types as answer. Please analyze the patch step by step, and then tell me which bug type. "
                       
    for commit, types in commits.items():
        if commit in result.keys():
            continue
        #print(commit)

        prompt = ""
        prompt += fix_prompt
        patch = get_patch(commit)
        prompt += "The given patch is: \""+patch+"\""
        if num_tokens_from_prompt(prompt)+512 > 4096:
            result[commit] = "TOO LONG(>4096)"
        else:
            passed_to_gpt += 1
            res = gpt4(prompt)
            result[commit] = res
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        print("correct/passed_to_gpt: "+str(correct)+"/"+str(passed_to_gpt))
        
def simple_ask(src_file,out_file):
    prompt="""I want you to act as a Linux kernel security patch expert, who is great at analyzing Linux security patches. A linux kernel patch is composed of commit title, message and diff.Commit diff is the differences for the patched files between the time when you are unpatched and the time when they are patched. I need your help to tell me the bug type of a give patch. I will give you commit diff and three lines around diff as context. I hope that you can first look at commit title and commit message, and then commit diff and its context; you can try to understand the root cause of the bugs fixed by the patch. There are seven possible bug types: use-after-free, memory out-of-bounds, memory leak, null pointer dereference, uninitilized values, race condition and others. You can choose one of bug types as answer. 

Memory-out-of-bounds occurs when the memory accessing is out of the valid range. Its common patch pattern is t o add boundary check and reset the size of the memory.
Use-after-free occurs when the used memory is already freed in another place, one common case is that the memory is freed in user-space. Its common patch pattern is to nullify the pointer or add lock/unlock operations.

I will pass you a patch, please tell me the bug type of the patch. If you think the bug type is not in above types, you can choose "others".
Please reply in the below json format; for example:
{
"bug type":"OOB/UAF/OTHERS",
"Analysis":""
}"""
    
    sleep_time = 5 
    #unknown_verified_patches_to_be_tested
    todo_commits={line.split(" ")[0]:line.split(" ")[1].strip() for line in helper.readFile(src_file)}
    tempature=1.0
    model="gpt-4-turbo"
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
        count = 0
        for commit, res in result.items():
            if todo_commits[commit] == "OTHER":
                continue
    
            if "\"bug type\":\""    in res :
                if "\"bug type\":\""  +todo_commits[commit]+"\"" in res:
                    count += 1
            elif "\"bug type\": \"" in res:
                if "\"bug type\": \""  +todo_commits[commit]+"\"" in res:
                    #print(commit)
                    count += 1
        print(count)
    else:
        result = dict()
    # exit()
    total = 0
    correct = 0
    
    fix_prompt = prompt
 
    repo_dir = linux_dir
    total = len(todo_commits)
    for commit in todo_commits:
        if commit in result:
            continue
      
        #print(commit)
        title_message = helper.get_commitMessage(repo_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
       
        # prompt = ""
        # prompt += fix_prompt + title_message + "\""
        prompt = ""
        prompt += fix_prompt + "\nTitle: "+title+"\n"
        prompt += "Commit message: "+message+""
        diff_3lines = helper.get_diff(repo_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        #print("commit: "+commit+"\nprompt: "+prompt)
        if prompt_len+512 > 128000:
            result[commit] = "TOO LONG(>128,000)"
        else:
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            retry = True
            while retry:
                try:
                    res = gpt_model(model,message,tempature)
                    
                except:
                    time.sleep(sleep_time)
                else:
                    retry = False
            # res = gpt_model(model,message,tempature)
            if "\"bug type\":\""+todo_commits[commit]+"\"" in res:
                correct += 1
            #print(res)
            print("correct/total: "+str(correct)+"/"+str(total))
            result[commit] = res
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        time.sleep(sleep_time)
 
        

def identify_functions(src_dir):
    fix_prompt = "I want you to act as a developer who has the bachelor's degree in computer engineerig and has proven experience in C program expert who is great at C programming, with a strong understanding of the language, syntax,"\
        "libries. Also, you are familiar with the Linux kernel and its source code. I will give you a piece of codes from Linux kernel, and the codes may be just one part of a function or several parts from different functions."\
        " I hope that you can identify all of functions inside it. The codes are : \""
    fix_prompt = "I want you to act as an expert who understands linux kernel functions greatly. I will give you a piece of code, and I hope that you give me all of functions inside the piece of code.  The codes are : \""
    
    out_file = HOME_DIR +"/results/all_funcs_in_finetune"+str(TEMPERATURE).replace(".", "")
    
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
        
    else:
        result = dict()
    
    for status in os.listdir(src_dir):
        print(status)
        for commit in os.listdir(src_dir+"/"+status):
            commit_dir = src_dir+"/"+status+"/"+commit+"/"
            for file in os.listdir(commit_dir):
                file_path = commit_dir+file
                if "after_patch/26d266e10e5e/drivers$mtd$nand$denali.c" in file_path:
                    continue
                if file_path in result.keys():
                    continue
                codes = ""
                for line in helper.readFile(file_path):
                    if line.startswith("\n"):
                        continue
                    codes += line[1:]
                prompt = ""
                prompt += fix_prompt
                prompt += codes +"\"."
                print(file_path)
                prompt_len = num_tokens_from_prompt(prompt)
                print(prompt_len)
                if prompt_len+512 > 4096:
                    result[file_path] = "TOO LONG(>4096)" #"there is no function call"
                else:
                    message = [{"role": "user", "content": prompt},{"role": "user", "content": "give me the functions in the format \" [func1, func2, func3...]\" "}]
                    res = gpt4(message,0)
                    result[file_path] = res
                    print(res)
                helper.del_file_if_exists(out_file)
                helper.dump(out_file, str(result))

def func_renaming(fix_prompt,src_file,out_file1,out_file2):
     
      
    if os.path.exists(out_file1):
        lines = helper.readFile(out_file1)
        result = ast.literal_eval(lines[0])
    else:
        result = dict()
    if os.path.exists(out_file2):
        lines = helper.readFile(out_file2)
        func_names_passed = ast.literal_eval(lines[0])
    else:
        func_names_passed = []    
    lines = helper.readFile(src_file)
    commit_file_funcs = ast.literal_eval(lines[0])
    func_lists = []
    for commit_file, funcs in commit_file_funcs.items():
        if "[" in funcs and "]" in funcs:
            curr_funcs = funcs.split("[")[1].split("]")[0].split(",")
            func_lists.extend(curr_funcs)
    func_lists = list(set(func_lists))
    func_lists = [func.strip() for func in func_lists]

    curr_funcs = []
    count = 0
    total = len(func_lists)
    for index in range(total):
        
        curr_func = func_lists[index]
        if curr_func == "":
            continue
        if curr_func in func_names_passed:
            continue
        count += 1
        
        func_names_passed.append(curr_func)
        curr_funcs.append(curr_func)
        if count == 20:
            print("index/total: "+str(index)+"/"+str(total))
            print("index, func_names_passed, total: "+str(index)+", "+str(len(func_names_passed))+", "+str(total))
            count = 0
            prompt = ""
            prompt += fix_prompt
            prompt += str(curr_funcs)+"\"."

            message = [{"role": "user", "content": prompt},{"role": "user", "content": "what are the new function names? in the format \"{given function name1: new_func_name1,given function name2: new_func_name2,given function name3: new_func_name3...}\", and the order of new function names should be same with the order of given function names"}]
            res = gpt4(message,0.7)
            print("passed funcs: "+str(curr_funcs))
            print("new names: "+str(res))
            result[str(curr_funcs)] = res
            curr_funcs = []
            helper.del_file_if_exists(out_file1)
            helper.dump(out_file1, str(result))
            helper.del_file_if_exists(out_file2)
            helper.dump(out_file2, str(func_names_passed))
            
def truncate_patch_diffs(src_file,out_file,prompt_file):
    repo_dir = linux_dir
    model = "gpt-4-turbo"
    temperature = 1.0
    max_tokens = 2048
     
    commits = [line.split(" ")[0].strip("\n") for line in helper.readFile(src_file) ]
    result = ast.literal_eval(helper.readFile(out_file)[0]) if os.path.exists(out_file) else {}
    for commit in commits:
        
        prompt = "".join(prompt_file)
            
        title_message = helper.get_commitMessage(repo_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
        
        # prompt = ""
        # prompt += fix_prompt + title_message + "\""
        prompt += title+"\n"
        prompt += message+"\n"
        
        diff_3lines = helper.get_diff1(repo_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "".join(diff_3lines)+"\""
        
        if num_tokens_from_prompt(message, model="gpt-4") +2048 > 128000:
            print("Prompt too long")
            return
        message = [{"role": "user", "content": prompt}]
        res = gpt4(message,0.7)
  
        print("response: {}\n".format(res))
        if "```diff" in res:
            diff = res.split("```diff")[1].split("```")[0]
        else:
            diff = res.split("```")[1].split("```")[-1]
        result[commit] = diff
        helper.delFileIfExists(out_file)
        helper.dump(out_file, str(result))

def parse(src_file,out_file):
    lines = helper.readFile(src_file)
    func_names = ast.literal_eval(lines[0])
    func_names_dict = dict()
    all_funcs = []
    for funcs, new_names in func_names.items():
        all_funcs.extend(ast.literal_eval(funcs))
        if "{" in new_names and "}" in new_names:
            temp = "{"+new_names.split("{")[1]
            try:
                func_names_dict.update(ast.literal_eval(temp))
            except:
                lines = temp.split("\n")
                for line in lines:
                    if ":" in line:
                        old_func = line.split(":")[0].strip().strip("\'").strip("\"")
                        new_func = line.split(":")[1].strip().strip(",").strip("\'").strip("\"")
                        func_names_dict[old_func] = new_func
    func_names_dict = {k: v for k, v in func_names_dict.items() if "NOT FUNCTION" not in v and "NOT_FUNCTION" not in v}
    func_names_dict1 = dict()
    for old_func, new_func in func_names_dict.items():        
        if "(" in old_func :
            old_func = old_func.split("(")[0]
        if " " in old_func:
            old_func = old_func.split(" ")[-1]
        func_names_dict1[old_func] = new_func

 

def llama_request(message,model,tempature,llama_key):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {llama_key}",
            # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
            # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
        },
        data=json.dumps({
            "model": model , # Optional
            "messages":message,
            "temperature":tempature,
            "max_tokens":2048,
        })
    )
    res = response.json()['choices'][0]['message']['content']
    return res

 
def bug_type_by_whole_patch_o1(groundtruth):
    out_file1 = HOME_DIR +"/xxx"
    out_file2 = HOME_DIR +"/xxx"
    result1= dict()
    result2 = dict()
    correct = 0
    total = 0
 
    todo = random.sample(list(groundtruth.keys()), 100)
    correct = 0
    index = 0
    fix_prompt = "".join(helper.readFile(HOME_DIR+"/prompts/reliable_classification"))
    for commit in todo:
        print("[step1]{}/{},{}".format(correct,index,commit))
        index += 1
        prompt = fix_prompt
        title_message = helper.get_commitMessage(linux_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        prompt += "\nCommit Title: "+title+"\n"
        prompt += "Commit message: "+message+""
        diff_3lines = helper.get_diff(linux_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
        # print(message)
        model = "o1-preview"
        tempature=1.0
        response = gpt_model_o1(model,message)
        result1[commit] = response
        #print(response)
        prompt = """
I will give you a response about if a patch contains reliable hints about the bug type. You are tasked with undertanding the response and conclude it in a json format; for example:
{
"contain reliable hints":"yes",
"bug type":"OOB"
}
or
{
"contain reliable hints":"no",
}
if the response is unclear about if a patch contains reliable hints about the bug type,
{
"contain reliable hints":"unclear"
}
Need to note that the bug type should be one of the following: use-after-free, out-of-bounds access, null pointer dereference, use before initilization, memory leak, missing permission check.
If the bug type does not fall into any of the above, you can just answer "other" for the "bug type" field.
The given response is as follows:
        """+response
        message = [{"role": "user", "content": prompt},]
        print("{},{}".format(index,commit))
        retry = True
        res = gpt_model_o1(model,message)
        result2[commit] = res
        helper.del_file_if_exists(out_file1)
        helper.del_file_if_exists(out_file2)
        helper.dump(out_file1, str(result1))
        helper.dump(out_file2, str(result2))
        time.sleep(5)
     
    
def is_reliable_llama(step1_out_file,step2_out_file):
    model = "meta-llama/llama-3.1-405b-instruct"
    tempature=1.0
    sleep_time = 5
    todo_commits = [line.split(" ")[0] for line in helper.readFile(HOME_DIR+"xxx")]
    if os.path.exists(step1_out_file):
        lines = helper.readFile(step1_out_file)
        result = ast.literal_eval(lines[0])
    else:
        result = dict()
    src_prompt = HOME_DIR+"prompts/reliable_classification"
    lines = helper.readFile(src_prompt)
    fix_prompt = "".join(lines)
 
    repo_dir = linux_dir
    total = len(todo_commits)
    index = 0
    for commit in todo_commits:
        if commit in result:
            continue
        index += 1
        print("[step1]{},{}".format(index,commit))
        title_message = helper.get_commitMessage(repo_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
       
        prompt = ""
        prompt += fix_prompt + "\nTitle: "+title+"\n"
        prompt += "Commit message: "+message+""
        diff_3lines = helper.get_diff(repo_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        #print("commit: "+commit+"\nprompt: "+prompt)
        if prompt_len+512 > 128000:
            result[commit] = "TOO LONG(>128,000)"
        else:
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            retry = True
            
            while retry:
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {llama_key}",
                            # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
                            # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
                        },
                        data=json.dumps({
                            "model": "meta-llama/llama-3.1-405b-instruct", # Optional
                            "messages":message,
                            "temperature":tempature,
                            "max_tokens":2048,
                        })
                    )
                    res = response.json()['choices'][0]['message']['content']
                except:
                    time.sleep(sleep_time)
                else:
                    retry = False
            #print(res)
            result[commit] = res
        helper.del_file_if_exists(step1_out_file)
        helper.dump(step1_out_file, str(result))
        time.sleep(sleep_time)
        
    if os.path.exists(step1_out_file):
        lines = helper.readFile(step1_out_file)
        result1 = ast.literal_eval(lines[0])
    else:
        result1 = dict()
    fix_prompt = "".join(helper.readFile(HOME_DIR+"/prompts/reliable_classification_step2"))
    index = 0
    for commit, step1response in result.items():
        if isinstance(step1response, list):
            continue
        index += 1
        prompt = fix_prompt+result[commit]
        message = [{"role": "user", "content": prompt},]
        print("[step2]: {},{}".format(index,commit))
        retry = True
        while retry:
            try:
                response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {llama_key}",
                            # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
                            # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
                        },
                        data=json.dumps({
                            "model": "meta-llama/llama-3.1-405b-instruct", # Optional
                            "messages":message,
                            "temperature":tempature,
                            "max_tokens":2048,
                        })
                    )
                res = response.json()['choices'][0]['message']['content']
            except:
                time.sleep(sleep_time)
            else:
                retry = False
        print(res)
        result1[commit] = [step1response,res]
        helper.del_file_if_exists(step2_out_file)
        helper.dump(step2_out_file, str(result1))
        time.sleep(sleep_time)
 
def is_reliable1(src_file,out_file,src_prompt):
    sleep_time = 5
    todo_commits = [line.strip("\n") for line in helper.readFile(src_file)]
    print("# of todo",len(todo_commits))
    tempature=1.0
    model="gpt-4-turbo"
    out_file = HOME_DIR +"xxx" 
 
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
       
    else:
        result = dict()

    total = 0
     
    lines = helper.readFile(src_prompt)
    fix_prompt = "".join(lines)
 
    repo_dir = linux_dir
    total = len(todo_commits)
    for commit in todo_commits:
        if commit in result:
            
            continue
        
        #print(commit)
        title_message = helper.get_commitMessage(repo_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
        
        prompt = ""
        prompt += fix_prompt + "\nTitle: "+title+"\n"
        prompt += "Commit message: "+message+""
        diff_3lines = helper.get_diff(repo_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        #print("commit: "+commit+"\nprompt: "+prompt)
        if prompt_len+512 > 128000:
            result[commit] = "TOO LONG(>128,000)"
        else:
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            retry = True
            while retry:
                try:
                    res = gpt_model(model,message,tempature)
                    
                except:
                    time.sleep(sleep_time)
                else:
                    retry = False
            # res = gpt_model(model,message,tempature)
            #print("commit: ", commit)
            #print(res)
            result[commit] = res
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        time.sleep(sleep_time)
def is_reliable2(src_file,out_file):
    sleep_time = 5
     
    tempature=1.0
    model="gpt-4-turbo"
 

    lines = helper.readFile(src_file)
    result = ast.literal_eval(lines[0])
    for commit, response in result.items():
        if isinstance(response, list):
            continue
        prompt = """
I will give you a response about if a patch contains reliable hints about the bug type. You are tasked with undertanding the response and conclude it in a json format; for example:
{
"contain reliable hints":"yes",
"bug type":"OOB"
}
or
{
"contain reliable hints":"no",
}
if the response is unclear about if a patch contains reliable hints about the bug type,
{
"contain reliable hints":"unclear"
}
Need to note that the bug type should be one of the following: use-after-free, out-of-bounds access, null pointer dereference, use before initilization, memory leak, missing permission check.
If the bug type does not fall into any of the above, you can just answer "other" for the "bug type" field.
The given response is as follows:
        """+result[commit]
        message = [{"role": "user", "content": prompt},]
        retry = True
        while retry:
            try:
                res = gpt_model(model,message,tempature)
                
            except:
                time.sleep(sleep_time)
            else:
                retry = False
        print("commit: ", commit)
        print(res)
        result[commit] = [response,res]
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        time.sleep(sleep_time)
          
def num_sentences(src_file):
    lines = helper.readFile(src_file)
    cveeval_commits =[x.strip("\n") for x in lines]
    fix_prompt = "I will give you the commit message of a patch commit. I want to know the number of sentences inside the commit message. But be noted that you need to exclude sentences, such as\"This is part of CVE-2019-3016.\"\
            ""(This bug was also independently discovered by Jim Mattson <jmattson@google.com>)\", which do not contain how the patch fix the bug or information about the bug itself.  The commit message of the patch is \""
    out_file = HOME_DIR +"/results/patch_num_sentences_"+str(TEMPERATURE).replace(".", "")
    
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
    else:
        result = dict()
 
    for commit in cveeval_commits:
        if commit in result.keys():
            continue
        title_message = helper.get_commitMessage(linux_dir,commit,True)
        message = " ".join(title_message[1:])
        
        if "c919a3069c77" == commit:
            message = "The gs_usb driver is performing USB transfers using buffers allocated on the stack. This causes the driver to not function with vmapped stacks. Instead, allocate memory for the transfer buffers."
        prompt = ""
        prompt += fix_prompt + message + "\""
        prompt_len = num_tokens_from_prompt(prompt)
        print("commit ",commit)
        if prompt_len+50 > 4096:
            result[commit] = "TOO LONG(>4096)"
        else:
            if "385aee965b4e" == commit or "4969c06a0d83" == commit or "52c479697c9b" == commit:
                result[commit] = 0
            elif "f2d67fec0b43" == commit or "dbcc7d57bffc" == commit or "f8d4f44df056" == commit:
                result[commit] = 30
            else:
                message = [{"role": "user", "content": prompt},{"role": "user", "content": "please only give the number of sentences inside the format [number]"}]
                res = gpt_model(message,1.0)
                
                print(res)
                result[commit] = int(re.findall(r'\d+', res)[0])
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
    return 2

#commits: to do commits
#out_file1: output file, which will be used by build_eval_data_for_random_given() of preprocess.py
def get_summaries(commits,out_file1):
    if os.path.exists(out_file1):
        lines = helper.readFile(out_file1)
        pairs = ast.literal_eval(lines[0])
    else:
        pairs = {}
    count = 0

    #commits=[line.split(" ")[1].split("id=")[1][:12] for line in helper.readFile(HOME_DIR+"/data/results/2024cves")]


    
    for commit in commits:
            
        if commit not in pairs.keys():
            count += 1
            print(commit)
            result = get_summary(commit).replace('\n', '')
            pairs[commit] = result
                
        helper.delFileIfExists(out_file1)
        helper.dump(out_file1, str(pairs))  
def is_reliable1(todo_commits,out_file,):
    # suffix = "_random500"
    sleep_time = 5
    # todo_commits = [line.strip("\n") for line in helper.readFile(HOME_DIR+"/data/cves/evaluation_set")]
    # todo_commits = [line.strip("\n") for line in helper.readFile(HOME_DIR+"/data/results/unknown_500_cases")]
    # todo_commits = [line.strip("\n") for line in helper.readFile(HOME_DIR+"/data/results/syzbot_50uafoob_filter_by_graphspd")]
    # todo_commits = [line.split(" ")[1].split("id=")[1] for line in helper.readFile(HOME_DIR+"/data/results/2024cves")]
    print("# of todo",len(todo_commits))
    tempature=1.0
    model="gpt-4-turbo"
    
    if os.path.exists(out_file):
        lines = helper.readFile(out_file)
        result = ast.literal_eval(lines[0])
    else:
        result = dict()

    total = 0
    keyword_match = 0
    
    src_prompt = HOME_DIR+"/codes/prompts/reliable_classification"
    lines = helper.readFile(src_prompt)
    fix_prompt = "".join(lines)
 
    repo_dir = linux_dir
    total = len(todo_commits)
    for commit in todo_commits:
        if commit in result:
            
            continue
        # if "fa40d9734a57" not in commit:
        #     continue
        print(commit)
        title_message = helper.get_commitMessage(repo_dir,commit,True)
        title = title_message[0].strip("\n").strip()
        message = [x.strip("\n").strip() for x in title_message[1:]]
        message = "".join(message)
        
        title_message = " ".join(title_message)
        if "9dc956b2c852" in commit:
            title_message = title_message.split("- Location")[0]
            print(title_message)
        # prompt = ""
        # prompt += fix_prompt + title_message + "\""
        prompt = ""
        prompt += fix_prompt + "\nTitle: "+title+"\n"
        prompt += "Commit message: "+message+""
        diff_3lines = helper.get_diff(repo_dir,commit)
        diff_3lines = [x for x in diff_3lines]
        prompt += "\n"+"Commit diff:\n"+"".join(diff_3lines)
        prompt_len = num_tokens_from_prompt(prompt)
        print("{}, prompt length: {}".format(commit, prompt_len))
        #print("commit: "+commit+"\nprompt: "+prompt)
        if prompt_len+512 > 128000:
            result[commit] = "TOO LONG(>128,000)"
        else:
            message = [{"role": "user", "content": prompt},]#{"role": "user", "content": "do the commit title and message contain the indications or implications of a security bug? please answer in the format of \"yes\" or \"no\"."}
            retry = True
            while retry:
                try:
                    res = gpt_model(model,message,tempature)
                    
                except:
                    time.sleep(sleep_time)
                else:
                    retry = False
            # res = gpt_model(model,message,tempature)
            #print("commit: ", commit)
            #print(res)
            result[commit] = res
        helper.del_file_if_exists(out_file)
        helper.dump(out_file, str(result))
        time.sleep(sleep_time)

# todo_commits=[] #a list of commits to be done
# out_file1="xxx" # output file of step 1 of llm 
# out_file2="xxx" # output file of step 2 of llm; which will be parsed to get the result
def llm_query(todo_commits,out_file1,out_file2):
    is_reliable1(todo_commits,out_file1)
    is_reliable2(out_file1,out_file2)
    
                                                                                   
