import math
import argparse
import json
import numpy as np
from tqdm import tqdm
# from sklearn.metrics import classification_report
from pathlib import Path
import sys
sys.path.append("../../")
import helper
import ast
import os
import random
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import collections

cwd = os.path.dirname(os.path.realpath(__file__))
linux_dir = str(Path(cwd).parent.absolute()) + "/repos/linux"
HOME_DIR = str(Path(cwd).parent.absolute())
PROJECT_DIR = str(Path(cwd).parent.parent.absolute())

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
    
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"FPRate: {avg_FPR:.1%}")
    print(f"FN Rate: {avg_FNR:.1%}")
    num_y_test = np.bincount(y_test)
    num_y_pred = np.bincount(y_pred)

    print(f"Ground Truth class counts: {num_y_test}")
    print(f"Predicted class counts: {num_y_pred}")
 

def get_metrics_binary(y_test,y_pred):
    y_test = [int(y) for y in y_test]
    y_pred = [int(y) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)

    # False positive rate (fall-out) is calculated as fp / (fp + tn)
    false_positive_rate = fp / (fp + tn)

    # False negative rate (miss rate) is calculated as fn / (fn + tp)
    false_negative_rate = fn / (fn + tp)

    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")
    print(f"False Positive Rate: {false_positive_rate}")
    print(f"False Negative Rate: {false_negative_rate}")

def get_groundtruth(type_num):
    if type_num == 3:
        return {line.split(": ")[0]:line.split(": ")[1].strip() for line in helper.readFile("xxx")  }
    elif type_num ==2:
        res = {} 
        for line in helper.readFile("xxx"):
            commit = line.split(": ")[0]
            label = line.split(": ")[1].strip()
            if "0" == label or "1" == label or "3" == label or "4" ==  label or "2" == label:
                res[commit] = "1"
            else:
                res[commit] = "0"#0 is non-mem; 1 is mem
        return res
    elif type_num == 23:
        res = {}
        for line in helper.readFile( "xxxx"):
            commit = line.split(": ")[0]
            
            label = line.split(": ")[1].strip()
            if "0" == label :
                res[commit] = "0"
            elif "2" ==  label:
                res[commit] = "1"
            elif "1" ==  label or "3" ==  label or "4" ==  label or "5" ==  label:
                res[commit] = "2"
        return res
    elif type_num == 12: 
        res = {}
        for line in helper.readFile("xxx"):
            commit = line.split(": ")[0]
            label = line.split(": ")[1].strip()
            if "6" ==  label :
                res[commit] = "0"
            else:
                res[commit] = "1"
        return res 
def get_reliable_conclusion(src_file,groundtruth1):
 
    import json
    json_str=helper.readFile(src_file)[0]
    reliable=[]
    not_reliable=[]
    reliable_pred_truth = collections.defaultdict(list)
    correct=0
    reliable_correct=0
    # print('9ea9b9c48387' in groundtruth1)
    for part in json_str.split("'], "):
        commit=part.split("'")[1][:12].strip()
  
        if "\"bug type\":[\"" in part:
            not_reliable.append(commit) 
        elif "contain reliable hints\":\"yes" in part or "contain reliable hints\": \"yes" in part:
            reliable.append(commit)
            bug_type = ""
            if "\"bug type\":\""  in part:
                bug_type = part.split("\"bug type\":\"")[1].split("\"")[0]
            if "\"bug type\": \""  in part:
                bug_type = part.split("\"bug type\": \"")[1].split("\"")[0]
            is_correct=False
            if "out-of-bound"   in bug_type:
                reliable_pred_truth[commit].append("0")
                reliable_pred_truth[commit].append(groundtruth1[commit])
            elif "use-after-free" in bug_type:
                reliable_pred_truth[commit].append("1")
                reliable_pred_truth[commit].append(groundtruth1[commit])
            else:
                reliable_pred_truth[commit].append("2")
                reliable_pred_truth[commit].append(groundtruth1[commit])
        elif "contain reliable hints\":\"no" in part or "contain reliable hints\": \"no" in part:
            not_reliable.append(commit)
        else:
            not_reliable.append(commit)
    
    print("reliable_correct: ", reliable_correct)
    print("reliable: ", len(reliable))
    print("not_reliable: ", len(not_reliable))
    return reliable, not_reliable,reliable_correct,reliable_pred_truth
def parse_reliable_classification(src_file ):
    json_str=helper.readFile(src_file)[0]
    reliable=[]
    not_reliable=[]
    reliable_result={}

    for part in json_str.split("'], "):
        commit=part.split("'")[1][:12].strip()
        # if commit not in groundtruth1:
        #     print("commit:-", commit)
        if "\"bug type\":[\"" in part:
            not_reliable.append(commit) 
        elif "contain reliable hints\":\"yes" in part or "contain reliable hints\": \"yes" in part:
            reliable.append(commit)
            bug_type = ""
            if "\"bug type\":\""  in part:
                bug_type = part.split("\"bug type\":\"")[1].split("\"")[0]
            if "\"bug type\": \""  in part:
                bug_type = part.split("\"bug type\": \"")[1].split("\"")[0]
            is_correct=False
            if "out-of-bound"   in bug_type:
                reliable_result[commit] = "OOB"
            elif "use-after-free" in bug_type:
                reliable_result[commit] = "UAF"
            else:
                 reliable_result[commit] = "OTHER"
            
        elif "contain reliable hints\":\"no" in part or "contain reliable hints\": \"no" in part:
            not_reliable.append(commit)
        else:
            not_reliable.append(commit)
    
    return reliable,reliable_result, not_reliable 
 
def get_reliable_classification_o1(src_file,pipeline_line_file,groundtruth_file):
    reliable,not_reliable=[], []
    commit_truth= {line.split(" ")[0]:line.split(" ")[1].strip() for line in helper.readFile(groundtruth_file)}
    pred,truth=[],[]
    for line in helper.readFile(src_file):
        commit = line.split("'")[1]
        if "nts\": \"yes" in line:
            reliable.append(commit)
            if "out-of-bound" in line:
                pred.append("0")
                truth.append(commit_truth[commit])
            elif "use-after-free" in line:
                pred.append("1")
                truth.append(commit_truth[commit])
            else:
                pred.append("2")
                truth.append(commit_truth[commit])
        else:
            not_reliable.append(commit)
    for line in helper.readFile(pipeline_line_file):
        commit = line.split(" ")[0]
        if commit in not_reliable:
            pred.append(line.split(" ")[1])
            truth.append(line.split(" ")[2].strip("\n"))
    get_metrics(truth,pred)
            

def reliable_classification1(src_file):
    groundtruth1 = get_groundtruth(3)
    groundtruth2 = get_groundtruth(23)
    count = 0
    reliable_correct = 0
    reliable_wrong = []
    one_type_count = 0
    not_reliable = []
    reliable = []
    pred_nonmem_true_nonmem = [line.split("\t")[0] for line in helper.readFile(HOME_DIR+"/data/results/mem_binary") if line.split("\t")[1]=="0" and "0" == line.split("\t")[2].strip("\n")]
    reliable_pred_truth = collections.defaultdict(list)
    
  
    reliable, not_reliable, reliable_correct,reliable_pred_truth = get_reliable_conclusion(src_file,groundtruth1)
    reliable_wrong=[]
    correct=0
    for commit in reliable_pred_truth:
        if reliable_pred_truth[commit][0] == reliable_pred_truth[commit][1]:
            correct+=1
    print("correct: ", correct)
    print("all: ", len(reliable_pred_truth))

    print("reliable_correct: ",reliable_correct)
    print("reliable: ", len(reliable))
    print("not_reliable: ", len(not_reliable))
    not_reliable_res=[]
    mem_correct = 0
    for commit in not_reliable:
         if commit in pred_nonmem_true_nonmem:
            mem_correct+=1
    print("mem_correct: ", mem_correct)
        
    

    mem_pred_truth = collections.defaultdict(list)
    notreliable_pred_truth = collections.defaultdict(list)
    threetype_pred_truth = collections.defaultdict(list)
    mem_truth=get_groundtruth(12)
    not_reliable1=[]
    for line in helper.readFile(HOME_DIR+"/data/results/mem_binary"):
        commit = line.split("\t")[0]
        
        if commit in not_reliable:
            mem_pred_truth[commit].append(line.split("\t")[1])
            mem_pred_truth[commit].append(mem_truth[commit])
            if mem_pred_truth[commit][1]=="0" and mem_pred_truth[commit][0]=="0":
                notreliable_pred_truth[commit].append("2")
                notreliable_pred_truth[commit].append("2")
            elif mem_pred_truth[commit][1]=="1" and mem_pred_truth[commit][0]=="1":
                not_reliable1.append(commit)
                # helper.dump(out_file, commit+"\n")
                notreliable_pred_truth[commit].append("1")
                notreliable_pred_truth[commit].append("1")
            elif mem_pred_truth[commit][1]=="0" and mem_pred_truth[commit][0]=="1":
                notreliable_pred_truth[commit].append("1")
                notreliable_pred_truth[commit].append("2")
            elif mem_pred_truth[commit][1]=="1" and mem_pred_truth[commit][0]=="0":
                notreliable_pred_truth[commit].append("2")
                notreliable_pred_truth[commit].append("1")
    
    correct=0
    for commit in mem_pred_truth:
        if mem_pred_truth[commit][0] == mem_pred_truth[commit][1]:
            correct+=1
    print("correct: ", correct)
    print("all: ", len(mem_pred_truth))

    res = []
    not_reliable_correct = 0
 
 
    eval_preds = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile("xxx") if "[" in line}
 
    count=0
    for commit, pred in eval_preds.items():
        
        commit = commit[:12]
        
        if commit in not_reliable:
            
            if notreliable_pred_truth[commit][0]=="1":
                notreliable_pred_truth[commit][0]=pred
                notreliable_pred_truth[commit][1]=groundtruth2[commit]
            elif notreliable_pred_truth[commit][1]=="1" and notreliable_pred_truth[commit][0]=="2":
                notreliable_pred_truth[commit][1]=groundtruth2[commit]                       
            percent = count/len(not_reliable)
            res.append(percent)
    for commit in not_reliable:
        if commit not in notreliable_pred_truth:
            notreliable_pred_truth[commit].append("2")
            notreliable_pred_truth[commit].append("2")
    correct=0
    for commit in notreliable_pred_truth:
        if mem_pred_truth[commit][0] == mem_pred_truth[commit][1]:
            if notreliable_pred_truth[commit][0] == notreliable_pred_truth[commit][1]:
                correct+=1
        if mem_pred_truth[commit][0] == mem_pred_truth[commit][1] and mem_pred_truth[commit][0] == "1":
            threetype_pred_truth[commit].append(notreliable_pred_truth[commit][0])
            threetype_pred_truth[commit].append(notreliable_pred_truth[commit][1])
    #     else:
    #         if mem_pred_truth[commit][0] == "0":
    #             notreliable_pred_truth[commit][0] = "2"
    def get_pred_truth(pred_truth):
        pred = []
        truth = []
        for commit in pred_truth:
            pred.append(pred_truth[commit][0])
            truth.append(pred_truth[commit][1])
        return pred, truth
    
 
    print("# of reliable: ", len(reliable_pred_truth))
    print("# of notreliable: ", len(notreliable_pred_truth))
    
    print("\nthe overll result for patches with hints: ")
    get_metrics(get_pred_truth(reliable_pred_truth)[0],get_pred_truth(reliable_pred_truth)[1])
    
    print("\nthe first step for patches without hints: ")
    get_metrics_binary(get_pred_truth(mem_pred_truth)[0],get_pred_truth(mem_pred_truth)[1])
    print("\nthe second step for patches without hints: ")
    get_metrics(get_pred_truth(threetype_pred_truth)[0],get_pred_truth(threetype_pred_truth)[1])
    print("\nthe overll result for patches without hints: ")
    get_metrics(get_pred_truth(notreliable_pred_truth)[0],get_pred_truth(notreliable_pred_truth)[1])
    
    print("\nthe overall results for all patches: ")
    get_metrics(get_pred_truth(notreliable_pred_truth)[0]+get_pred_truth(reliable_pred_truth)[0],get_pred_truth(notreliable_pred_truth)[1]+get_pred_truth(reliable_pred_truth)[1])
 
    print("all: ", len(notreliable_pred_truth))
  
      
  
def parse_random_results(src_file,res_file1,res_file2):
    reliable,reliable_results, not_reliable = parse_reliable_classification(src_file)

    print("reliable: ", reliable_results)
    print(len(reliable))
    print("not_reliable: ", len(not_reliable))
    
    eval_preds = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile(res_file1) if "[" in line and  "\t" in line}
    for commit, res in reliable_results.items():
        if res=="UAF" or res=="OOB":
            print("https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id="+commit+";"+res)
    print("\n\n---\n\n")
    eval_preds1 = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile(res_file2) if "[" in line and  "\t" in line}
    for commit in not_reliable:
        if eval_preds1[commit] == "1":
            if eval_preds[commit] == "0":
                print("https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id="+commit+";OOB")
            elif eval_preds[commit] == "1":
                print("https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id="+commit+";UAF")         
  
           
def eval_results_given(commit_truth,src_file, is_selected,commits):
 
    eval_preds = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile(src_file) if "[" in line and  "\t" in line}
    preds, truths = [],[]
    for commit, pred in eval_preds.items():
        if is_selected:
            if commit in commits:
                preds.append(pred)
                truths.append(commit_truth[commit])
        else:
            preds.append(pred)
            truths.append(commit_truth[commit])
    
    print(len(preds))
    get_metrics(preds, truths)
 
def draw_confusion_matrix(y_true, y_pred,fig_name):
    labels =  ["OOB","UAF", "OTHER",]
    cm = confusion_matrix(y_true, y_pred, labels=["0", "1", "2",])
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Convert confusion matrix to DataFrame (for better visualisation)
    cm_df = pd.DataFrame(cm, index=["0", "1", "2"], columns=["0", "1", "2"])
    annotations = cm_df.applymap(lambda x: f"{x*100:.2f}%")

    # Visualise confusion matrix using seaborn's heatmap
    # plt.figure(figsize=(10,7))
    plt.figure(figsize=(12,8))
    ax = sns.heatmap(cm_df, annot=annotations, fmt='', cmap='Blues', cbar=False,  annot_kws={"size": 27})
    ax.set_yticklabels(labels, fontsize=22, rotation=60, ha="right")
    ax.set_xticklabels(labels, fontsize=22)
    ax.set_yticklabels(labels, fontsize=22)
    plt.ylabel('Actual',fontsize=22 , y=0.55, va='center')
    plt.xlabel('Predicted',fontsize=22)
    plt.tight_layout()
    # plt.subplots_adjust(right=0.2)
    plt.savefig(HOME_DIR+"/results/figs/"+fig_name, format='pdf')
    
def get_llm_conclusion(src_file):
    json_str=helper.readFile(src_file)[0]
    reliable=[]
    not_reliable=[]
    llm_conclusion = {}
 
    llm_res={}
    for part in json_str.split("'], "):
        commit=part.split("'")[1][:12].strip()
     
        if "\"bug type\":[\"" in part:
            not_reliable.append(commit) 
        elif "contain reliable hints\":\"yes" in part or "contain reliable hints\": \"yes" in part:
            reliable.append(commit)
            bug_type = ""
            llm_res[commit]=part
            if "\"bug type\":\""  in part:
                bug_type = part.split("\"bug type\":\"")[1].split("\"")[0]
            if "\"bug type\": \""  in part:
                bug_type = part.split("\"bug type\": \"")[1].split("\"")[0]
            is_correct=False
            if "out-of-bound"   in bug_type:
                llm_conclusion[commit]="OOB"
                 
            elif "use-after-free" in bug_type:
                llm_conclusion[commit]="UAF"
            else:
                llm_conclusion[commit]="OTHER"
            # if not is_correct:
            #     print("commit: ", commit," bug_type: ", bug_type, " groundtruth: ", groundtruth1[commit])
                
            # elif "\"bug type\": \"" not in part  :
            #     print("commit: ", commit)
        elif "contain reliable hints\":\"no" in part or "contain reliable hints\": \"no" in part:
            not_reliable.append(commit)
        else:
            not_reliable.append(commit)
    return reliable, not_reliable,llm_conclusion

#src_file: out_file2 in llm_query() of llm_query.py
def parse_llm_results(src_file):
    #reliable are commits which are classified as patches with hints by LLM
    #then you can get the bug type for these commits by retriving from llm_conclusion
    #not_reliable are commits which are classified as patches without hints by LLM, which should be passed to the next step (SliceLM)
    #reliable, not_reliable, llm_conclusion = parse_reliable_classification(src_file)
    reliable, not_reliable, llm_conclusion = get_llm_conclusion(src_file)
    
    out_str = ""
    for commit in reliable:
        out_str += commit + " : " + llm_conclusion[commit] + "\n"
    return not_reliable, out_str
    
 #not_reliable: commits to be processed by SliceLM
 #res_file1: the result file from the first step (SliceLM)
 #res_file2: the result file from the second step (SliceLM)   
def parse_sliceLM_results(not_reliable,res_file1,res_file2):

    out_str=""
    eval_preds = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile(res_file1) if "[" in line and  "\t" in line}
    eval_preds1 = {line.split("\t")[2]:line.split("\t")[0] for line in helper.readFile(res_file2) if "[" in line and  "\t" in line}
    for commit in not_reliable:
        if eval_preds1[commit] == "1":
            if eval_preds[commit] == "0":
                out_str+=commit+" : OOB\n"
            elif eval_preds[commit] == "1":
                out_str+=commit+" : UAF\n"
            else:
                out_str+=commit+" : OTHER\n"
        else:
            out_str+=commit+" : OTHER\n"
    return out_str
                