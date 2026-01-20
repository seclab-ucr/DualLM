import subprocess
import shutil
import datetime
import re
import os
import json
import requests
from bs4 import BeautifulSoup
from requests.api import patch
import glob
import subprocess
import requests
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import sys
from diskcache import Cache
linuxDir = "/xxx/xxx/xxxx/linuxs/linux-stable"
base_url = 'https://elixir.bootlin.com/linux/'
# url = urljoin(base_url, 'A/ident/sscanf')
base_url2 = 'https://elixir.bootlin.com/linux/'
cache_dir = "cache"
  
def get_current_time():
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M:%S")
    return current_time

def dump(fileName,str):
    f = open(fileName,"a")
    f.write(str)
    f.close()

def delFileIfExists(path):
    if os.path.exists(path):
        os.remove(path)
   
        
def del_file_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def readFile(fileName):
    f = open(fileName,"r")
    contents = f.readlines()
    f.close()
    return contents

def num2percent(num1,num2):
    return str(round(num1/(num2),4)*100)+"%"

def _trim_lines(buf):
    for i in range(len(buf)):
        if len(buf[i])==0:
            continue
        if buf[i][-1] == '\n':
            buf[i] = buf[i][:-1]

def get_patchedPatches(file):
    f = open(file,"r")
    patchInfo = f.readlines()
    f.close()
    patched = []
    for line in patchInfo:
        linelist=line.split("$")
        (linuxCommit,bug_url,bug_name)=(linelist[0],linelist[2],linelist[3])
        patched.append(linuxCommit+"$"+bug_name.strip("\n")+"$"+bug_url)
    return patched
    
def get_patchesWithFixes():
    f = open(patches_fixes,"r")
    lines = f.readlines()
    hashes = []
    patches = []
    for line in lines:
        if line == "\n":
            continue
        patches.append(line.split("-")[0])
        hashes.append(line.split("-")[1])
    return hashes
def get_currentHash(work_dir):
    cmd = "cd "+work_dir+";git rev-parse HEAD"
    res = command(cmd)
    return res[0]
def get_commitMessage(dir, commitnumber, onlyMessage):
    """Get commit message for a specific commit.
    
    Args:
        dir (str): Repository directory
        commitnumber (str): Commit hash
        onlyMessage (bool): If True, return only the commit message without metadata
        
    Returns:
        list: Lines of the commit message
        
    Raises:
        RuntimeError: If the commit cannot be found or other git errors occur
    """
    try:
        cmd = f"cd {dir};git log --format=%B -n 1 {commitnumber} --pretty=fuller"
        result = command(cmd)
        
        if not result:
            raise RuntimeError(f"No commit message found for commit: {commitnumber}")
            
        temp = result[6:]  # Skip the first 6 lines which contain git metadata
        res = []
        
        if onlyMessage:
            for line in temp:
                # Stop at metadata lines
                if any(line.startswith(prefix) for prefix in [
                    "    Signed-off-by",
                    "    Reported-by",
                    "    Fixes",
                    "    Tested-by",
                    "Cc:"
                ]):
                    break
                res.append(line)
        else:
            res = temp
            
        if not res:
            print(f"Warning: Empty commit message for commit {commitnumber}")
            
        return res
        
    except RuntimeError as e:
        raise RuntimeError(f"Failed to get commit message for {commitnumber}: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while getting commit message for {commitnumber}: {str(e)}")

def get_commitMessage_ffmpeg(dir,commitnumber,onlyMessage):
    cmd="cd "+dir+";git log --format=%B -n 1 "+commitnumber+" --pretty=fuller"
    result = command(cmd)
    temp = result[6:]
    res = []
    if onlyMessage:
        for line in temp:
            if line.startswith("    Signed-off-by") == True or line.startswith("    Found-by")==True  \
            or line.startswith("    Found-by") == True :
                break
            res.append(line)
    else:
        for line in temp:
            res.append(line)
    return res

  
def get_commitLog(dir,logAll):
    if logAll:
        cmd="cd "+dir+";git log --all --oneline"
    else:
        cmd="cd "+dir+";git log --oneline"
    return command(cmd)

def command(cmd):
    try:
        # Run command and capture both stdout and stderr
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        
        # Check return code
        if p.returncode != 0:
            error_msg = stderr.decode('utf-8', errors='replace')
            raise RuntimeError(f"Command failed with exit code {p.returncode}.\nCommand: {cmd}\nError: {error_msg}")
        
        # Process output
        result = stdout.splitlines()
        res = []
        for line in result:
            if b"d354d9afe923 [PATCH] fbcon: don" in line:
                res.append("[PATCH] fbcon: don<B4>t call set_par() in fbcon_init() if vc_mode == KD_GRAPHICS")
            elif b"+    /* search for the magic dword - '_SM_" in line and b"as DWORD formatted -  on paragraph boundaries */" in line:
                res.append("+    /* search for the magic dword - '_SM_b4 as DWORD formatted -  on paragraph boundaries */\n")
            else:
                try:
                    res.append(line.decode("ISO-8859-1"))
                except UnicodeDecodeError:
                    print(f"Warning: UnicodeDecodeError in command: {cmd}")
                    print(f"Problematic line: {line}")
                    # Try UTF-8 as fallback
                    res.append(line.decode('utf-8', errors='replace'))
        
        return res
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"Failed to execute command: {cmd}\nError: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error while executing command: {cmd}\nError: {str(e)}")

def run_cmd_return_status(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    return process.returncode

def run_cmd_return_status1(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process.wait()
    return process.returncode,process.stdout.read()

def show_mergedCommits(kernel,commit):
    cmd="cd "+kernel+";git show -1 "+commit
    #p=subprocess.Popen(string1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p_buf=command(cmd)
    for line in p_buf:
        if line.startswith("Merge"):
            break
    line=line[:-1]
    linelist=line.split(" ")
    (commit1,commit2)=(linelist[1],linelist[2])
    if is_commitHashLegal(kernel,commit1) == False or is_commitHashLegal(kernel,commit2) == False:
        return "NOTMERGE"
    cmd = "cd "+kernel+";git checkout "+commit1
    command(cmd)
    cmd = "cd "+kernel+";git log --oneline "+commit1
    res1 = command(cmd)
    cmd = "cd "+kernel+";git checkout "+commit2 
    res2 = command(cmd)
    cmd = "cd "+kernel+";git log --oneline "+commit2
    res2 = command(cmd)
    startpoint = -1
    for index in range(len(res2)):
        if res2[index] in res1:
            if res2[index+1] == res1[res1.index(res2[index])+1] and res2[index+2] == res1[res1.index(res2[index])+2]:
                startpoint = index
                break
    return (res2[:startpoint])

def get_when_merge(direcory,commit):
    cmd = "cd "+direcory+";git find-merge "+commit+" master"
    res = command(cmd)
    if res == []:
        return commit
    if "Commit is directly on this branch" in res[0]:
        return commit
    else:
        return res[0].strip().split(" ")[-1]

def get_mergedCommits(kernel,commit):
    cmd="cd "+kernel+";git show -1 "+commit
    #p=subprocess.Popen(string1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p_buf=command(cmd)
    for line in p_buf:
        if line.startswith("Merge"):
            break
    line=line[:-1]
    linelist=line.split(" ")
    (commit1,commit2)=(linelist[1],linelist[2])
    return linelist[2]

def get_fixesHash(commitMessage):
    for line in commitMessage:
        line = str.encode(line)
        if line.startswith(b"    Fixes:") and b"Fixes:\n" not in line:
            #print(line)
            res = line.split(b" ")[5]
            if b"(" in res:
                res = res.split(b"(")[0] 
            res = res.decode("utf-8")    
            return res
def extract_title(commitMessage):
    res = b"NOTFOUND404"
    for index in range(len(commitMessage)):
        line = commitMessage[index]
        line = str.encode(line)
        if line.startswith(b"    Fixes:"):
            # print("line: "+str(line))
            if b"(" in line:
                res = line.split(b"(")[1].split(b")")[0]
                if b")" not in line:
                    line1 = commitMessage[index+1]
                    if line1.startswith("    Fixes:") == False or line1.startswith("    Signed-off-by") == False or line1.startswith("    Reported-by")== False:
                        res = res.strip(b"\n") + b" "+ str.encode(line1.split(")")[0].strip("    "))
                    print("res: "+str(res))
    res = res.decode("utf-8")
    #print("res")
    return res.strip("\"")
def containFixesTag(linuxDir,commit):
    message = get_commitMessage(linuxDir,commit,False)
    for line in message:
        if line.startswith("    Fixes:") and "Fixes:\n" not in line:
            return True
    return False
def containCCStable(commit):
    message = get_commitMessage(linuxDir,commit,False)
    for line in message:
        if line.startswith("    Cc:") and "stable@" in line:
            return True
    return False
def containCCStable1(directory,commit):
    message = get_commitMessage(directory,commit,False)
    for line in message:
        if line.startswith("    Cc:") and "stable@" in line:
            return True
    return False



def contains(commitHash,tag):
    cmd="cd "+linuxDir+";git branch --contains "+commitHash
    result=command(cmd)
    print(result)
    if result == []:
        return False
    for r in result:
        print(r)
        if tag  in r:
            return True
    return False

def get_author(directory,commit):
    cmd = "cd "+directory+";git show "+commit
    message = command(cmd)
    for line in message:
        if line.startswith("Author:"):
            author=line[8:].strip("\n")
            return author
    return ""
def get_metrics(y_test,y_pred):
    # Assuming y_test is your ground truth labels and y_pred are the predicted labels.
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # For multi-class, you will need to set the "average" parameter.
    # Options include 'micro', 'macro', 'weighted', 'samples', or None.
    average = 'weighted'

    precision = metrics.precision_score(y_test, y_pred, average=average)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred, average=average)
    recall = metrics.recall_score(y_test, y_pred, average=average)

    # False Negative Rate is calculated as FN / (TP + FN)
    # For multi-class, we calculate per class and then take the average.
    FN = cm.sum(axis=1) - np.diag(cm)  
    TP = np.diag(cm)
    FNR = FN / (TP + FN+ np.finfo(float).eps)
    FP = cm.sum(axis=0) - np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    FPR = FP / (FP + TN)
    # To get the average FNR:
    avg_FNR = np.mean(FNR)
    # To get the average FPR:
    avg_FPR = np.mean(FPR)

    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"Average False Negative Rate: {avg_FNR}")
    print(f"Average False Positive Rate: {avg_FPR}")
    num_y_test = np.bincount(y_test)
    num_y_pred = np.bincount(y_pred)

    print(f"Ground Truth class counts: {num_y_test}")
    print(f"Predicted class counts: {num_y_pred}")
def get_commitAuthor(directory,commit):
    cmd = "cd "+directory+";git log --format=%B -n 1 "+commit+" --pretty=fuller"
    message = command(cmd)
    for line in message:
        if line.startswith("Commit:"):
            author=line[8:].strip("\n").strip()
            return author.split(" <")[0]
    return ""

def get_signedoff(message):
    signedoffs = []
    for line in message:
        line = line.strip("\n")
        if line.startswith("    Signed-off-by"):
            if ": " in line:
                signedoffs.append(line.split(": ")[1])
            else:
                signedoffs.append(line.split(":")[1])
    return signedoffs

def get_commit(c_buf):
    for line in c_buf:
        if line.startswith("commit"):
            commit=line[6:]
            commit.strip()
            return commit
    return ""
def switch_mon(month):
    switcher = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12
    }
    return switcher.get(month, None)
#Data example: "Date:   Wed Oct 7 10:55:41 2015 -0700\n"
#commitdate example "CommitDate: Mon Aug 8 17:29:06 2016 -0700\n"
#return value example:  datetime.datetime(2015, 10, 7, 10, 55, 41)
def get_time(Date):
    if not Date:
        return None
    Date=Date.split(" ")
    month=switch_mon(Date[2])
    day=int(Date[3])
    hour= int(Date[4].split(":")[0])
    minute=int(Date[4].split(":")[1])
    second=int(Date[4].split(":")[2])
    year=int(Date[5])
    time=datetime.datetime(year,month,day,hour,minute,second)
    return time

def get_commitDate(c_buf):
    for line in c_buf:
        if line.startswith("CommitDate"):
            #print line
            return line

def get_authorDate(c_buf):
    for line in c_buf:
        if line.startswith("AuthorDate"):
            #print line
            return line

def get_commitTime(directory,commitnumber):
    string1="cd "+directory+";git show --pretty=fuller "+commitnumber
    s_buf=command(string1)
    _trim_lines(s_buf)
    commitDate=get_commitDate(s_buf)
    committime=get_time(commitDate)

    return committime
def get_authorTime(directory,commitnumber):
    string1="cd "+directory+";git show --pretty=fuller "+commitnumber
    s_buf=command(string1)
    _trim_lines(s_buf)
    commitDate=get_authorDate(s_buf)
    committime=get_time(commitDate)
    return committime

def get_bugDetails(commit):
    patchInfo = readFile(patches)
    for line in patchInfo:
        line = line[:-1][1:-1]
        linelist = line.split(", ")
        commitnumber, bug_url,bug_name = linelist[3][1:-1],linelist[5][1:-1],linelist[6][1:-1]
        if commitnumber == commit:
            return bug_name+","+bug_url+","+commitnumber
def get_bugUrl(name):
    name=""
    patchInfo = readFile(patches)
    for line in patchInfo:
        line = line[:-1][1:-1]
        linelist = line.split("\', ")

        bug_url = linelist[5][1:-1]
        bug_name = linelist[6][1:-1]
        if name.strip("\n") == bug_name.strip("'"):
            return bug_url    
def has_cc(dir,commit):
    commitMessage = get_commitMessage(dir,commit,False)
    for line in commitMessage:
         if line.startswith(b"    Cc:"):
             if "stable@vger.kernel.org" in line:
                 print(line+" "+commit)
                 return True
    return False

def checkout(directory, tag):
    cmd = "cd "+directory+"; git checkout -f "+tag
    return command(cmd)   

def get_HashTitles(tag,startcommit):
    cmd = "cd "+linuxDir+";git checkout -f "+tag
    command(cmd)
    cmd = "cd "+linuxDir+";git log "+startcommit+".. --oneline"
    res = command(cmd)
    hash_titles = dict()
    for line in res:
        hash_titles[line[13:].strip("\n")] = line[:12]
    return hash_titles
def get_HashTitleDict(tag,startcommit,directory):
    cmd = "cd "+directory+";git checkout -f "+tag
    command(cmd)
    cmd = "cd "+directory+";git log "+startcommit+".. --oneline"
    res = command(cmd)
    hash_titles = dict()
    for line in res:
        hash_titles[line[:12]] = line[13:].strip("\n")
    return hash_titles
def copy_LinuxTemp():
    cmd = "cd "+linuxDir+";cd ..;cp -r linux-stable linux-temp"
    command(cmd)
def delete_LinuxTemp():
    cmd = "cd "+linuxDir+";cd ..;rm -rf linux-temp"
    command(cmd)
#give a str like, v4.14.60,return v4.14.59
#note: only for vx.x.x version format currently
def get_previousTag(directory,tag):
    cmd = "cd "+directory+";git describe --abbrev=0 "+tag+"^"
    res = command(cmd)
    return res[0].strip("\n")
    #return "v"+(".").join(tag.split(".")[:-1])+"."+str(int(tag.split(".")[-1])-1)



def extractSHA1FromLine(line1,line2):
    res1 = re.findall(r"\b[0-9a-f]{5,40}\b",line1)
    if res1 == []:
        res1 = re.findall(r"\b[0-9a-f]{5,40}\b",line2)
    return res1

def extractSHA(message):
    p_upstream = re.compile(r'commit ([0-9a-f]+) upstream', re.I)
    p_upstream2 = re.compile(r'upstream commit ([0-9a-f]+)', re.I)
    p_upstream3 = re.compile(r'Upstream commit ([0-9a-f]+)', re.I)
    m = p_upstream.search(message) or p_upstream2.search(message) or p_upstream3.search(message)
    if m:
        return m.group(1)
    else:
        return ""

def title2hash(directory,title):
    cmd = "cd "+directory+";git checkout -f master"
    command(cmd)
    cmd = "cd "+directory+";git log --oneline|grep \""+str(title).replace("\"","\\\"")+"\""
    res = command(cmd)
    if res == []:
        return "NOTFOUND"
    commit = res[0][:12]
    if is_commitHashLegal(directory,commit) == False:
        return "NOTFOUND"
    return res[0][:12]

def get_commitContents(directory,commit,withContext):
    contents = []
    cmd = "cd "+directory+";git show -m "+commit
    result = command(cmd)
    contents_start = False
    for line in result:
        if line.startswith("diff"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") or line.startswith("@@ ") or line.startswith("+++") or line.startswith("---"):
                continue
            if withContext == False:
                if line.startswith("-") == False and line.startswith("+") == False:
                    continue
                contents.append(line)
            else:
                contents.append(line)
    return contents

def get_diff(directory,commit):
    contents = []
    cmd = "cd "+directory+";git show -m "+commit
    result = command(cmd)
    contents_start = False
    for line in result:
        if line.startswith("diff"):
            contents_start = True
        if contents_start:
            if line.startswith("diff") or line.startswith("index ") :
                continue
            contents.append(line)
    return contents

def get_diff1(directory, commit):
    cmd = "cd "+directory+";git show "+commit
    result = command(cmd)
    diff_start = False
    res = []
    for line in result:
        if line.startswith("diff"):
            diff_start = True
        if diff_start:
            res.append(line)
    return res

def get_diff2(directory, commit):
    cmd = "cd "+directory+";git show "+commit
    result = command(cmd)
    diff_start = False
    res = []
    for line in result:
        if line.startswith("-") or line.startswith("+"):
            res.append(line)
    return res


def get_contentFileDict(directory,commit):
    cmd = "cd "+directory+";git show "+commit
    result = command(cmd)
    rvalue = dict()
    for line in result:
        line = line.strip("\n")
        if line.startswith("diff"):            
            changeFile = line.split(" b/")[1]
            rvalue[changeFile] = []
        if line.startswith("-") == False and line.startswith("+") == False:
            continue
        if line.startswith("---") == True or line.startswith("+++") == True:
            continue
        rvalue[changeFile].append(line)
    return rvalue

def Merge(dict1, dict2):
    return (dict2.update(dict1))

def get_LTSCommits(directory,lts):
    cmd = "cd "+directory+";git checkout -f remotes/origin/linux-"+lts+".y"
    command(cmd)
    cmd = "cd "+directory+";git log v"+lts+".. --oneline"
    res = command(cmd)
    hash_titles = dict()
    for line in res:
        hash_titles[line[:12]] = line[13:].strip("\n")
    return hash_titles

def is_commitSame(dir1,dir2,commit1,commit2,withContext):
    content1 = get_commitContents(dir1,commit1,withContext)
    content2 = get_commitContents(dir2,commit2,withContext)
    if len(content1) != len(content2):
        return False
    for index in range(len(content1)):
        if content1[index] != content2[index]:
            #print(content1[index])
            return False
    return True
def is_commitHashLegal(directory,commit):
    if len(commit) < 8:
        return False
    cmd = "cd "+directory+"; git show "+commit
    result = command(cmd)
    if "fatal: ambiguous argument" in result[0] or "fatal: bad object" in result[0]:
        return False
    return True
def get_normalDay(day):
    if "0:00:00" in day:
        return 0
    if ":" in day:
        if "-" in day:
            return -1
        return 1
    return int(day)
#get all commits between startcommit and endcommit
#return a dict,the key is commit title, the vaule is commit hash
def get_hashtitleDictOfRange(directory,startcommit,endcommit):
    title_hash = dict()
    cmd = "cd "+directory+";git checkout -f "+endcommit
    command(cmd)
    cmd = "cd "+directory+"; git log --oneline "+startcommit+".."+endcommit
    result = command(cmd)
    for line in result:
        title_hash[line[:12]] = line[13:].strip("\n")
    return title_hash
#show the commit hash of a tag
def tag2commit(tag):
    cmd = "cd "+linuxDir+";git rev-parse "+tag+"^{}"
    result = command(cmd)
    return result[0]
#get all commits within a tag
#give a tag, like v4.14.190,return all commits between v4.14.189 and v4.14.190
def get_commitsWithinTag(tag):
    tagsplit = tag.split(".")
    index = int(tagsplit[-1])-1
    if index == 0:
        previoustag = ".".join(tagsplit[:-1])
    else:
        previoustag = ".".join(tagsplit[:-1])+"."+str(index)
    startcommit = tag2commit(previoustag).replace("\n","")
    endcommit = tag2commit(tag).replace("\n","")

    hash_title = get_hashtitleDictOfRange(linuxDir,startcommit,endcommit)
    return hash_title
#given a commit and a repo branch, try to get the corresponding commit in main branch.
def get_maincommit(repopath,branch,commit):
    string1='cd '+repopath+';git rev-list '+commit+'..'+branch+' --ancestry-path'
    resultlist1=command(string1)
    string1='cd '+repopath+';git rev-list '+commit+'..'+branch+' --first-parent'
    resultlist2=command(string1)
    print("get_maincommit: mainline extracted: "+commit)
    commoncommitlist = [commit for commit in resultlist1 if commit in resultlist2]
    return commoncommitlist[-1][:12]
def get_Title(directory,commitHash):
    cmd="cd "+directory+";git show --oneline "+commitHash
    result=command(cmd)
    title = result[0][13:].strip(" ").strip("\n")
    return title
def get_moduleFromTitle(title):
    module = title.split(":")[0]
    return module
def calculateDelay(dir1,dir2,commit1,commit2):
    date1 = get_commitTime(dir1,commit1)
    date2 = get_commitTime(dir2,commit2)
    # print("data1: "+str(date1)+" commit1: "+commit1+" dir1: "+dir1)
    # print("date2: "+str(date2)+" commit2: "+commit2+" dir2: "+dir2)
    # print("str(date2-date1): "+str(date2-date1))
    days = str(date2-date1).split(",")[0].split(" ")[0]
    #print(days)
    return days
#True if commit1 is before commit2
#False if commit1 is after commit2
def relativeOrderOfTwoCommits(direcory,commit1,commit2):
    cmd = "git rev-list --count "+commit1+".."+commit2
    result1 = command(cmd)
    cmd = "git rev-list --count "+commit2+".."+commit1
    result2 = command(cmd)
    if int(result1[0]) > 0 and int(result2[0])==0:
        return True
    return False

def get_priorCommit(direcory,commit):
    cmd = "cd "+direcory+";git show "+commit+"^ --oneline"
    result = command(cmd)
    return result[0][:12]
def is_cherrypickedbackported(directory,commit):
    message = get_commitMessage(directory,commit,False)
    for line in message:    
        if "(backported from commit" in line or "(cherry picked from commit" in line or "(cherry-picked from commit" in line or "(cherry-picked from" in line or "(cherry picked from" in line or "back-ported from" in line or "backport to" in line or "back ported from" in line or "backported from" in line:
            # print(line)
            return True
    return False

def get_mainlineCommitForPicked(direcory,commit):
    message = get_commitMessage(direcory,commit,False)
    for line in message:    
        if "(backported from commit" in line or "(cherry picked from commit" in line or "(cherry-picked from commit" in line or "(cherry-picked from" in line or "(cherry picked from" in line or "back-ported from" in line or "backport to" in line or "back ported from" in line or "backported from" in line:
            if "http" in line:
                return ["NOTFOUND"]
            return re.findall(r"\b[0-9a-f]{5,40}\b",line)
    return ["NOTFOUND"]
    
def onlyDate(date):
    date = str(date).split(" ")[0]
    date = datetime.datetime.strptime(date,"%Y-%m-%d")
    return date

def diffNum(commit,directory):
    cmd = "cd "+directory+";git show "+commit
    result = command(cmd)
    num = 0
    for line in result:
        if line.startswith("diff"):
            num += 1
    return num

def get_changedFile(commit,directory):
    cmd = "cd "+directory+";git show "+commit
    result = command(cmd)
    files = []
    for line in result:
        if line.startswith("diff"):
            files.append(line[13:].split(" ")[0])
    return files
def get_commitFiles(work_dir,commit, chosen_symbol):
    lines = get_commitContents(work_dir,commit, False)
    files = []
    for line in lines:
        if line.startswith("diff"):
            if chosen_symbol == "-":
                file_path = line.split("--git a/")[1].split(" b/")[0]
                file = file_path.split("/")[-1]
                # print(f"{file_path} {file}")
            else:
                file = line.split("/")[-1].strip("\n")
            files.append(file)
    return files
def parse_logline(line):
    return line[:12],line[13:].strip("\n")
def get_changedFuncs(commit, directory):
    cmd = "cd "+directory+";git show "+commit+"| grep -E '^(@@)' | sed 's/@@.*@@//'"
    res = command(cmd)
    funcs = []
    for line in res:
        pieces = line.split("(")
        if len(pieces) > 1:
            funcs.append(pieces[0].split(" ")[-1])
    return funcs
def get_commitsChageFile(file,direcory):
    cmd = "cd "+direcory+";git log --oneline --follow -- "+file
    res = command(cmd)
    rvalue = []
    for line in res:
        rvalue.append(line[:12])
    return rvalue
def contain_fixesTag(directory,commitnumber):
    res = get_commitMessage(directory,commitnumber,False)
    for line in res:
        if line.startswith("    Fixes:"):
            if "https://" in line or "http://" in line:
                return False
            if len(res) >=res.index(line) + 2 and res[res.index(line) + 1].startswith("    -"):
                return False
            if line == '    Fixes:\n':
                return False
            return True
    return False
def get_fixtag(commit,directory):
    fixtagline = re.compile(r'^\s+(Fixes):.*$', re.I)
    p_fixes = re.compile(r'^\s+Fixes:\s+(commit\s+)?<?([0-9a-f]{6,})>?.*$')
    res = get_commitMessage(directory,commit,False)
    for line in res:
        if fixtagline.match(line):
            if "http://" in line or "https://" in line:
                continue
            if "Fixes: v" in line:
                continue
            m = p_fixes.match(line)
            if m:
                return m.group(2)
    return ""
def version_compare(version1, version2):
    cmd = "printf \'"+version1+"\n"+version2+"\n\'|sort -V"
    res = command(cmd)
    if version1 in res[0]:
        return True
    else:
        return False
def version_tag_correct(commit, lts):
    message = get_commitMessage(linuxDir,commit,False)
    version_reg = re.compile(r'.*#\s*.*v?([0-9]\.[0-9]{1,2}\.?[0-9]{0,2}).*', re.I)
    for mess_line in message:
        if mess_line.startswith("    Cc:") and "stable@" in mess_line:
            if "#" in mess_line:
                res = version_reg.match(mess_line)
                if res:
                    version = res.group(1)
                    if version.startswith("6."):
                        if "+" in mess_line:
                            return True
                        else:
                            return False
                    else:
                        if lts in version:
                            return True
                        else:
                            if version_compare(version, lts):
                                if "up" in mess_line or "+" in mess_line or "above" in mess_line:
                                    return True
            return False
                
def get_version(work_dir,commit):
    cmd = "cd "+work_dir+";git describe --contains "+commit
    res = command(cmd)
    return res[0].split("~")[0]

def get_contain_branch(directory, commit):
    cmd = "cd "+directory+";git branch --contains "+commit
    return command(cmd)
def contain_fixesTagWithReturnedLine(commitnumber,direcory):
    res = get_commitMessage(direcory,commitnumber,False)
    for line in res:
        if line.startswith("    Fixes:"):
            if "https://" in line or "http://" in line:
                return ""
            if len(res) >=res.index(line) + 2 and res[res.index(line) + 1].startswith("    -"):
                return ""
            if line == '    Fixes:\n':
                return ""
            return line
    return ""

def contain_commit(directory,commitHash,tag):
    cmd="cd "+directory+";git branch --contains "+commitHash
    result= command(cmd)
    #content = (commitHash,result)
    #dump(resultOfContainsCheckInEachCommit,str(content)+"\n")
    if result == []:
        return False
    for r in result:
        if tag in r:
            return True
    return False
def get_revertedTitle(title):
    res = title.replace("Revert \"","")[:-1]
    return res

def orderDictByValue(dict):
    return sorted(dict.items(), key=lambda x: x[1], reverse=True) 

def commitIsBefore(maybe_ancestor_commit,descendant_commit,directory):
    cmd = "cd "+directory+";git merge-base --is-ancestor "+maybe_ancestor_commit+" "+descendant_commit+";echo $?"
    res = command(cmd)[0].strip("\n")
    if res == "0":
        return True
    if res == "1":
        return False


def countElementInDict(dict1,key):
    if key not in dict1.keys():
        dict1[key] = 1
    else:
        dict1[key] += 1
    return dict1
def ymd2Datetime(ymd):
    datetime1 = ymd.split("-")
    baseTime = datetime.datetime(int(datetime1[0]),int(datetime1[1]),int(datetime1[2]),0,0,0)
    return baseTime



def get_lts_commits(LTSbranch,startcommit):
    branch = "remotes/origin/linux-"+LTSbranch+".y"
    cmd = "cd "+linuxDir+";git checkout "+branch
    command(cmd)
    cmd = "cd "+linuxDir+";git log --oneline "+startcommit+".."
    res = command(cmd)
    return res

def is_reported_syzbot(dir,commitnumber):
    cmd="cd "+dir+";git log --format=%B -n 1 "+commitnumber+" --pretty=fuller"
    result = command(cmd)
    temp = result[6:]
    res = []
    for line in temp:
        if line.startswith("    Reported-by")==True or line.startswith("    Tested-by") == True or line.startswith("    Reported-and-tested-by") == True:
            if "syzbot" in line or "syzkaller" in line:
                return True
    return False



#def downstreamTitle2upstreamHash()    
commoncommitlist_null =[]

def get_mainCommit(repopath,branch,commit):
    string1='cd '+repopath+';git rev-list '+commit+'..'+branch+' --ancestry-path'
    resultlist1=command(string1)
    string1='cd '+repopath+';git rev-list '+commit+'..'+branch+' --first-parent'
    resultlist2=command(string1)
    commoncommitlist = [commit for commit in resultlist1 if commit in resultlist2]
    if commoncommitlist == []:
        commoncommitlist_null.append(commit)
        return commit
    return commoncommitlist[-1][:12]

