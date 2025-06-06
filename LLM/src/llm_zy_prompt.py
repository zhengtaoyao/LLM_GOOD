from openail.utils import efficient_openai_text_api, set_endpoints, openai_text_api, openai_text_api_with_top_p, load_partial_openai_result, save_partial_openai_result, retrieve_dict, compute_ece, plot_calibration_curve, openai_text_api_with_backoff, num_tokens_from_string
from helper.data import get_dataset, inject_random_noise_y_level
from helper.args import get_command_line_args
from helper.active import train_lr, inference_lr
from helper.utils import load_yaml
from openail.config import configs
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import ast
from openail.utils import load_mapping
import ipdb
import os.path as osp
import editdistance
from collections import Counter
import random
import re
import string
import numpy as np
from models.nn import LinearRegression
from helper.utils import noise_transition_matrix
import seaborn as sns
import ipdb 
from helper.train_utils import calibration_plot
import pandas as pd
import sys

pal = sns.color_palette("crest")
sns.set_palette(pal)

# Set the figure size
plt.figure(figsize=(12, 12))  # 12 inches by 8 inches

# Set global text sizes
plt.rcParams['font.size'] = 20         # Default font size
plt.rcParams['axes.titlesize'] = 20    # Axes title size
plt.rcParams['axes.labelsize'] = 20    # X and Y axes label size
plt.rcParams['xtick.labelsize'] = 20   # X-tick label size
plt.rcParams['ytick.labelsize'] = 20   # Y-tick label size
plt.rcParams['legend.fontsize'] = 20   # Legend font size
plt.rcParams['font.family'] = 'sans-serif' # Figure title size

colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")


def most_common_number(numbers):
    counter = Counter(numbers)
    most_common = counter.most_common(1)  # Get the most common number and its count
    return most_common[0][0]  # Return the most common number


def get_closest_label(input_string, label_names):
    min_distance = float('inf')
    closest_label = None
    for label in label_names:
        distance = editdistance.eval(input_string, label)
        if distance < min_distance:
            min_distance = distance
            closest_label = label
    return closest_label

def keep_first_n_words(paragraph, n):
    words = paragraph.split()
    first_512_words = ' '.join(words[:n])
    return first_512_words


class Experiment:
    def __init__(self, data, api_key, data_path) -> None:
        self.raw_texts = data.raw_texts
        self.label_names = data.label_names
        self.category_names = data.category_names
        self.api_key = api_key
        self.data_path = data_path
        self.num_of_node = len(self.raw_texts)

    def load_cache(self, dataset, prompt_key):
        cache = load_partial_openai_result(self.data_path, dataset, prompt_key)
        return cache

    def save_cache(self, dataset, prompt_key, res):
        save_partial_openai_result(self.data_path, dataset, res, prompt_key, load_pre_existing=None, num_of_elements=self.num_of_node)

    def sync_api(self, prompts, temperature = None, top_p = None, n = 1, dataset = 'cora', key = 'zero_shot', rewrite = False):
        assert (temperature != None or top_p != None), "one of them must be set" 
        responses = []
        cache = self.load_cache(dataset, key)
        if cache != None:
            cache = cache.get(key, None)
        for i, prompt in tqdm(enumerate(prompts)):
            if prompt == "":
                responses.append("")
                continue
            if cache != None and cache[i] != "" and not rewrite:
                responses.append(cache[i])
            else:
                if temperature != None:
                    res = openai_text_api_with_backoff(prompt, self.api_key, temperature=temperature, n = n)
                else:
                    res = openai_text_api_with_top_p(prompt, self.api_key, top_p=top_p, n = n)
                if not self.check_grammar(res['choices'][0]['message']['content']) and key != 'consistency':
                    res = self.fix_grammar(res['choices'][0]['message']['content'])
                    time.sleep(3)
                responses.append(res)
                time.sleep(3)
        return responses

    def test_no_correction(self, prompts, label_names, temperature = None, top_p = None, n = 1, dataset = 'cora', key = 'zero_shot', rewrite = False, fix = False, seed = 0, sp = 2, ss = 2):
        responses = []
        errors = 0
        valid_number = len([x for x in prompts if x != ""])
        input_filename = "no_correct_prompt_async_input_{}_{}_temperature_{}_n_{}_input_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        output_filename = "no_correct_prompt_async_input_{}_{}_temperature_{}_n_{}_output_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        openai_result = efficient_openai_text_api(prompts, input_filename, output_filename, sp, ss, api_key=self.api_key, temperature=temperature, n = n, rewrite = rewrite)
        for i, res in tqdm(enumerate(openai_result)):
            if i not in select_ids:
                responses.append("")
                continue
            try:
                if key != 'consistency' and 'all' not in key:
                    check_res, error_type, error_message = self.check_grammar(res[0][0])
                    if check_res:
                        responses.append(res[0])
                        continue
                else:
                    check_res, error_type, error_message = self.check_grammar(res[0][0])
                    if check_res:
                        responses.append(res[0])
                        continue
            except Exception:
                check_res = False
                error_type = 'grammar error'
                error_message = None
                responses.append('')
                errors += 1
        print("Error number: {}".format(errors))
        return responses, None


    def async_api(self, prompts, label_names, temperature=None, top_p=None, n=1, dataset='cora', key='zero_shot', rewrite=True, fix=False, seed=0, sp=2, ss=2):
        responses = []
        select_ids = [i for i in range(len(prompts)) if prompts[i] != ""]
        valid_number = len(select_ids)
        input_filename = "prompt_async_input_{}_{}_temperature_{}_n_{}_input_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        output_filename = "prompt_async_input_{}_{}_temperature_{}_n_{}_output_seed_{}_{}.json".format(dataset, key, temperature, n, seed, valid_number)
        
        openai_result = efficient_openai_text_api(
            prompts, 
            input_filename, 
            output_filename, 
            sp, 
            ss, 
            api_key=self.api_key, 
            temperature=temperature, 
            n=n, 
            rewrite=rewrite
        )
        
        for i, res in tqdm(enumerate(openai_result)):
            if i not in select_ids:
                responses.append("")
                continue
            # Directly append the response without any grammar check or correction
            responses.append(res)
            # Optionally, you can adjust the sleep time if necessary
            # time.sleep(3)
        
        # Since we're not fixing any grammar, error index list is empty
        return responses, []


        
    def vanilla_verbalized_confidence(self, question):
        return question
    


    
    def topk_confidence(self, question, k=1, name='arxiv'):
        if k == 1:
            return "Question: {}. Provide your answer to the question. For example,  \
    [{{\"answer\": <your_answer>, \"Confidence\": <Your confidence here>, \"Reason\": <Your reason here>}}]".format(question)
        elif name == 'arxiv':
            return "Paper: {}. Which arxiv CS-subcategory does this paper belong to? Output your answer in the form of arXiv CS sub-categories like \"cs.XX\" together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{{\"answer\":<answer_here>, \"confidence\": <confidence_here>}}, ...]\n".format(question)
        else:
            return "Question: {}. Provide your {} best guesses and a confidence number that each is correct \
    (0 to 100) for the question from most probable to least. The sum of all confidence should be 100. For example,  \
    [{{\"answer\": <your_first_answer>, \"confidence\": <confidence_for_first_answer>}}, ...]        \
    ".format(question, k)

    def retrieve_multiple_answers(self, answer, data, num):
        output = []
        invalid = 0
        for result in answer:
            if result == "":
                res = [("", 0)]
                output.append(res)
                continue
            res = []
            for r in result:
                line = r.lower()
                this_line = []
                try:
                    ## if no error, retrieve all dicts in a list
                    this_dict = retrieve_dict(line)
                    for dic in this_dict:
                        answer = dic['answer']
                        confidence = dic['confidence']

                        if isinstance(confidence, str):
                            # If it includes a '%', remove it
                            if '%' in confidence:
                                confidence = confidence.replace('%', '')
                            try:
                                confidence = float(confidence)
                            except ValueError:
                                # If still not valid, set a default or skip
                                confidence = 0.0

                        this_line.append((answer, confidence))
                    res.append(this_line)
                except:
                    ## if error, split the result based on }, 
                    parts = line.split("},")
                    for p in parts:
                        try: 
                            ans = get_closest_label(p, self.label_names)
                            confidence = max(int(''.join(filter(str.isdigit, p))), 100)
                        except Exception:
                            confidence = 0
                        this_line.append((ans, confidence))
                        invalid += 1

                    res.append(this_line)
            output.append(res)
        print("invalid number: {}".format(invalid))
        return output
    
    def retrieve_answer(self, answer, data):
        output = []
        invalid = 0
        for result in answer:

            if result == "":
                res = [("", 0)]
                output.append(res)
                continue
            if isinstance(result[0], list):
                # If result[0] is a list, extract the first string
                line = result[0][0].lower()
            elif isinstance(result[0], str):
                # If result[0] is a string, process it directly
                line = result[0].lower()
            else:
                # If result[0] is neither a string nor a list, skip it
                output.append([("", 0)])
                invalid += 1
                continue

            
            
            try:
                this_dict = retrieve_dict(line)
                res = []
                for dic in this_dict:
                    answer = dic['answer']
                    confidence = dic['confidence']
                    
                    # Ensure confidence is a float
                    if isinstance(confidence, str) and '%' in confidence:
                        confidence = confidence.replace('%', '')
                    try:
                        confidence = float(confidence)
                    except ValueError:
                        confidence = 0.0
                    
                    res.append((answer, confidence))
                output.append(res)

            except Exception:
                # Handle 'none' as OOD
                if 'none' in line:
                    res = [('none', 100)]
                else:
                    answer = get_closest_label(line, self.label_names)
                    res = [(answer, 100)]  # Default confidence to 100 for k=1
                    invalid += 1
                output.append(res)
                continue
        print("invalid number: {}".format(invalid))
        return output




    def save_result(self, num_of_nodes, select_ids, pred, conf, data_obj, dataset_name = 'cora', seed = 0, strategy = 'random', method = 'zero_shot'):
        y_pred = torch.tensor([-1 for _ in range(num_of_nodes)])
        y_conf = torch.tensor([0. for _ in range(num_of_nodes)])
        y_pred[select_ids] = pred
        y_conf[select_ids] = conf
        res = {
            'pred': y_pred,
            'conf': y_conf,
            'test_mask': data_obj.test_masks[seed]
        }
        torch.save(res, osp.join(self.data_path, 'active', '{}^result^{}^{}^{}.pt'.format(dataset_name, strategy, method, seed)))

    def load_saved(self, dataset_name, prompt_key):
        cache_path = osp.join(self.data_path, 'active', '{}^active^{}.pt'.format(dataset_name, prompt_key))

        if osp.exists(cache_path):
            cache = torch.load(cache_path)
            return cache
    
    def save_saved(self, dataset_name, prompt_key, new_y_pred, new_y_conf, select_ids, total_num):
        cache_path = osp.join(self.data_path, 'active', '{}^active^{}.pt'.format(dataset_name, prompt_key))

        num_nodes = total_num
        if osp.exists(cache_path):
            cache = torch.load(cache_path)
            y_pred, y_conf = cache['pred'], cache['conf']
            if len(y_pred) != num_nodes:
                y_pred = torch.tensor([-1 for _ in range(num_nodes)])
                y_conf = torch.tensor([0. for _ in range(num_nodes)])
        else:
            y_pred = torch.tensor([-1 for _ in range(num_nodes)])
            y_conf = torch.tensor([0. for _ in range(num_nodes)])
        y_pred[select_ids] = new_y_pred
        y_conf[select_ids] = new_y_conf  

        res = {
            'pred': y_pred,
            'conf': y_conf,
        }


        torch.save(res, cache_path)

    def fix_grammar(self, old):
        if '[' not in old or ']' not in old: return ""
        start = old.find('[')
        end = old.find(']', start) + 1  # +1 to include the closing bracket
        old = old[start:end]
        prompt = "Extract a valid python object from the following text, just output the processed object, do not output anything else. \
Old one: {} \n New one here:".format(old)
        new_res = openai_text_api(prompt, self.api_key, temperature=0, n = 1)
        return new_res
    



    def check_grammar(self, old, format = '[]'):
        clean_t = old
        list_str = ""
        if format == '[]':
            start = clean_t.find('[')
            end = clean_t.find(']', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
        else:
            start = clean_t.find('{')
            end = clean_t.find('}', start) + 1  # +1 to include the closing bracket
            list_str = clean_t[start:end]
        list_str = list_str.lower()
        try:
            result = ast.literal_eval(list_str)
        except Exception:
            return False, 'grammar error', None
        try:
            first_answer = result[0]
            if not isinstance(first_answer, dict):
                return False, 'grammar error', None
            else:
                answer = first_answer['answer']
                confidence = first_answer['confidence']

                if answer in self.label_names:
                    return True, 'success', None
                else:
                    return False, 'format error', answer
        except Exception:
            return False, 'format error', None
     

    def eval_result(self, res, label_names, select_ids, gt, dataset_name='cora', method='zero_shot', strategy='random', data_obj=None, seed=0, g_error_idx=None):
        pred = []
        conf = []
        cannot_fix = 0
        gt_y = gt[select_ids]
        print(label_names)
        
        for idx in select_ids:
            r = res[idx]
            if r == "":
                pred.append(-1)  # Mark as unfixable
                conf.append(0.0)  # No confidence
                continue  # Skip if the response is empty
            
            # Process the response 'r' for different methods
            if 'consistency' in method or 'all' in method:
                # For consistency methods, 'r' contains multiple responses
                this_pred = []
                this_conf = []
                for ans in r:
                    p = ans[0]
                    c = ans[1] if len(ans) > 1 else 100  # Default confidence to 100 if not provided
                    
                    if p == 'none':
                        # Handle OOD case by assigning a special label (-2)
                        p = -2
                    elif p in label_names:
                        p = label_names.index(p)
                    else:
                        # Attempt to fix the label using closest match
                        p_fixed = get_closest_label(p, label_names)
                        p = label_names.index(p_fixed)
                        c = c / 2  # Reduce confidence for incorrect formats
                        cannot_fix += 1
                    
                    this_pred.append(p)
                    this_conf.append(c)
                
                # Majority voting among predictions
                p_final = most_common_number(this_pred)
                indices = [i for i, x in enumerate(this_pred) if x == p_final]
                # Average confidence for the majority-voted predictions
                c_final = sum([this_conf[i] for i in indices]) / len(indices)
                pred.append(p_final)
                conf.append(c_final)
            
            else:
                # Single response case (e.g., zero-shot or k=1)
                ans = r[0]
                p = ans[0]
                c = ans[1] if len(ans) > 1 else 100  # Default confidence to 100 if missing
                
                if p == 'none':
                    # Handle OOD case
                    p = -2
                elif p in label_names:
                    p = label_names.index(p)
                else:
                    # Fix label if it's invalid
                    p_fixed = get_closest_label(p, label_names)
                    p = label_names.index(p_fixed)
                    c = c / 2
                    cannot_fix += 1
                
                pred.append(p)
                conf.append(c)
        
        # Convert predictions and confidences to tensors
        pred = torch.tensor(pred)
        conf = torch.tensor(conf) / 100.0  # Normalize confidence values to [0, 1]
        
        # Map ground truth OOD labels to -1 if not already mapped
        gt_y_mapped = gt_y.clone()
        
        # Compute accuracy metrics
        all_acc = (pred == gt_y_mapped).float().mean()
        # Filtered accuracy: only consider predictions with confidence > 0
        filter_mask = conf > 0
        filter_acc = (pred[filter_mask] == gt_y_mapped[filter_mask]).float().mean()
        
        # Compute Expected Calibration Error (ECE)
        filter_conf = conf[filter_mask]
        filter_pred = pred[filter_mask]
        filter_label = gt_y_mapped[filter_mask]
        ece = compute_ece(filter_conf, filter_pred, filter_label, n_bins=10)
        
        # Save results
        self.save_result(len(gt), select_ids, pred, conf, data_obj, dataset_name, seed, strategy, method)
        total_num = len(gt)
        self.save_saved(dataset_name, method, pred, conf, select_ids, total_num)
        
        print("Cannot fix number: {}".format(cannot_fix))
        print(f"All Accuracy: {all_acc * 100:.2f}%, Filtered Accuracy: {filter_acc * 100:.2f}%, ECE: {ece:.2f}, Number of Labels: {len(filter_label)}")
        
        return all_acc, filter_acc, ece






# # #below is short new prompt for ood
def zero_shot_prompt(texts, label_names, need_tasks=True, object_cat="Paper", question="Given the current possible classes, determine if it belongs to one of them. If so, specify that class; otherwise, say \"none\".", answer_format="Provide your answer in the form of a python dictionary like [{\"answer\": <class_name_or_'none'>, \"confidence\": <confidence>}]."):

    prompts = []
    for text in texts:
        prompt = "{}: \n".format(object_cat)
        prompt += (text + "\n")
        prompt += "As a research scientist, your task is to analyze and classify {object_cat}s based on their main topics, meanings, background, and methods. Please first read the content of the {object_cat} carefully. And then identify the {object_cat}'s key focus. Finally match the content to one of the given classes.\n"
        if not 'arxiv' in question:
            prompt += "There are following classes: \n"
            prompt += "[" + ", ".join(label_names) + "]" + "\n"
        prompt += question + "\n"
        #current no need task!!! so the answer format is not used!
        if need_tasks:
            prompt += answer_format
        prompts.append(prompt)
    return prompts



# # # #below is long new prompt for ood finally used in the paper llm-good

# def zero_shot_prompt(texts, label_names, need_tasks=True, object_cat="Paper", question="Given the current possible classes, determine if it belongs to one of them. If so, specify that class; otherwise, say \"none\".", answer_format="Provide your answer in the form of a python dictionary like [{\"answer\": <class_name_or_'none'>, \"confidence\": <confidence>}]."):
#     prompts = []
#     for text in texts:
#         prompt = (
#             f"You are an expert text classification assistant specializing in identifying whether a given {object_cat} belongs to the predefined in-distribution categories or is out-of-distribution (OOD).\n"

#             # f"You are an intelligent and professional assistant that detects out-of-distribution (OOD) {object_cat} in text data.\n"
#             f"A {object_cat} is considered as **out-of-distribution ** if it does NOT belong to **any of the in-distribution category(ies)** listed below.\n"
#             # f"## Task:\n"
#             f"Your task is, given the content of the {object_cat} below, to determine whether it is an out-of-distribution (OOD) {object_cat}. If it is an OOD {object_cat}, answer \"none\", if it is not an OOD {object_cat}, determine which in-distribution category below it belongs to. Provide a brief explanation "
#             f"of your reasoning and assign a confidence score between 0 and 1 for your justification.\n\n"
#             f"## In-distribution Categories:\n"
#             # f"### Normal Category(ies):\n"
#         )

#         for category in label_names:
#             prompt += f"- **{category}**\n"

#         prompt += (
            
#             f"If you are uncertain whether the {object_cat} **significantly aligns** with **any of the in-distribution category(ies)**, assume that it does NOT align with them, which means it is an out-of-distribution {object_cat}.\n"
#             f"\The description of the {object_cat} that you need to identify is as follows:\n"
#             f"{text}\n"
#         )

#         prompts.append(prompt)
#     return prompts









def few_shot_prompts(texts, label_names, example, example_label, object_cat = "Paper", question = "Which arxiv cs subcategories does this paper belong to?", answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]"):
    prompts = []
    for text in texts:
        prompt = "I will first give you an example and you should complete task following the example.\n"
        prompt += zero_shot_prompt([example], label_names, need_tasks = True, object_cat = object_cat, question = question, answer_format = answer_format)[0]
        prompt += "\nOutput:\n[{{\"answer\":\"{}\", \"confidence\":{}}}]\n".format(example_label, 100)
        prompt += zero_shot_prompt([text], label_names, need_tasks = True, object_cat = object_cat, question = question, answer_format = answer_format)[0]
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts

def few_shot_topk_prompts(texts, label_names, example, example_dict, exp, object_cat = "Paper", question = "Which arxiv cs subcategories does this paper belong to?", answer_format = "Give 3 likely arXiv CS sub-categories as a comma-separated list ordered from most to least likely together with a confidence ranging from 0 to 100, in the form of a list python dicts like [{\"answer:\":<answer_here>, \"confidence\": <confidence_here>}]", name = 'arxiv'):
    prompts = []
    for text in texts:
        prompt = "I will first give you an example and you should complete task following the example.\n"
        in_context = example + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following classes: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
            in_context += question + "\n"
        prompt += exp.topk_confidence(in_context, 3, name)
        prompt += "\nOutput:\n"
        prompt += example_dict + "\n"
        in_context = text + "\n"
        in_context += "Task: \n"
        if not 'arxiv' in question:
            in_context += "There are following classes: \n"
            in_context += "[" + ", ".join(label_names) + "]" + "\n"
        in_context += question + "\n"
        prompt += exp.topk_confidence(in_context, 3, name)
        prompt += "\nOutput:\n"
        prompts.append(prompt)
    return prompts 


def calculate_cost_from_a_list_of_texts(texts, select_ids = None):
    cost = 0
    for i, text in enumerate(texts):
        if select_ids != None and i not in select_ids: continue
        if len(text) > 0:

            cost += num_tokens_from_string(text)

    return cost



def pack_prompt(prompts, select_idx, number):
    new_prompts = ["" for _ in range(number)]
    for i, prompt in enumerate(prompts):
        if i in select_idx:
            new_prompts[i] = prompt
    return new_prompts





def consistency(topk_prompt, dataname, exp, temperature=1.2, data=None, strategy='random', seed=0, key_name='consistency'):
    n = 1
    topk_response, error_idx = exp.async_api(topk_prompt, data.label_names, temperature, None, n=n, dataset=dataname, key=key_name, rewrite=False, fix=True, seed=seed, sp=60, ss=1.5)

    exp.save_cache(dataname, key_name, topk_response)
    res = exp.retrieve_answer(topk_response, data)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method=key_name, strategy = strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
    response_str = []
    for x in topk_response:
        try:
            if x != '':
                response_str.append(x[0] + x[1] + x[2])
        except Exception:
            response_str.append("")
    return eval_out, response_str




def consistency_no_topk(prompt, dataname, exp, temperature = 1.2, data = None, strategy = 'random', seed = 0, key_name = 'consistency_no_topk'):
    ## 
    n = 3
    response, error_idx = exp.async_api(prompt, data.label_names, temperature, None, n = n, dataset = dataname, key = key_name, rewrite = False, fix = True, seed = seed, sp = 60, ss = 1.5)
    exp.save_cache(dataname, key_name, response)
    res = exp.retrieve_multiple_answers(response, data, n)
    eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method=key_name, strategy = strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
    response_str = []
    for x in response:
        try:
            if x != '':
                response_str.append(x[0] + x[1] + x[2])
        except Exception:
            response_str.append("")
    return eval_out, response_str



def clean_header(text):
    text = re.sub(r'(From:\s+[^\n]+\n)', '', text)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)

    return text

def clean_text(text):  
    re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')      
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    
    return text




def count_and_plot(tensor, tensor2, xtick_labels=None, output_name = "", title="Cora original sampled label distribution"):
    plt.figure(figsize=(12, 6))
    unique_labels, counts = torch.unique(tensor, return_counts=True)
    unique_labels2, counts2 = torch.unique(tensor2, return_counts=True)

    hue = ['Ground truth' for _ in range(len(unique_labels2))] + ['Annotations' for _ in range(len(unique_labels))]

    x = unique_labels.tolist() + unique_labels2.tolist()

    x = [str(a) for a in x]

    counts = counts.tolist() + counts2.tolist()

    df = pd.DataFrame({'Labels': x, '#Appearance': counts, 'Type': hue})

    cmap_pal = sns.color_palette("deep")

    sns.set_palette(cmap_pal)

    ax = sns.barplot(data=df, x="Labels", y="#Appearance", hue="Type")

    plt.legend(loc='upper right', fontsize=20)

    plt.ylim(0, 375)

    plt.xticks(fontsize=35, fontweight='bold')

    # Adjust y-axis label properties
    plt.yticks(fontsize=35, fontweight='bold')

    plt.xlabel('Labels', fontsize=35, fontweight='bold')

    plt.ylabel('#Appearances', fontsize=35, fontweight='bold')

    plt.tick_params(axis='x', which='both', length=0)
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.tick_params(axis='x', which='both', length=0)

    legend = ax.legend()
    legend.set_title('')
    plt.setp(legend.get_title(), visible=False)


    plt.tight_layout()
    plt.savefig(output_name)
    plt.clf()





if __name__ == '__main__':
    print("LLM GOOD")
    args = get_command_line_args()    
    params_dict = load_yaml(args.yaml_path)

    key = params_dict['OPENAI_KEY']
    data_path = params_dict['DATA_PATH']

    seeds = [i for i in range(args.main_seed_num)]
    seeds = [0]

    if args.dataset == '':
        need_datasets = ['citeseer', 'cora', 'wikics','pubmed']
    else:
        need_datasets = [args.dataset]

    need_datasets = ['cora']

    if args.filter_strategy == 'none':
        exps = ['draw']
    else:
        exps = [args.filter_strategy]
    
    for dataname in need_datasets:
        reliability_list = []
        params_dict = load_yaml(args.yaml_path)
        data_path = params_dict['DATA_PATH']
        data = get_dataset(
    seeds,
    dataname,
    split='no',            # Set split to 'no' or any non-active value
    data_format=args.data_format,
    data_path=data_path,
    logit_path=None,
    random_noise=0,
    no_val=1,
    budget=args.budget,
    strategy='no',         # Set strategy to 'no' to skip node selection
    num_centers=0,
    compensation=0,
    save_data=0,
    llm_strategy='none',
    max_part=0,
    oracle_acc=1,
    reliability_list=[],
    total_budget=-1,       # total_budget will be set in data.py
    second_filter='none',
    train_stage=True,
    post_pro=False,
    filter_all_wrong_labels=False,
    alpha=args.alpha,
    beta=args.beta,
    gamma=args.gamma,
    ratio=args.ratio
)

        # Full mapping of class names to indices for each dataset
        full_mapping = {
            'cora': {
                'Rule_Learning': 0,
                'Neural_Networks': 1,
                'Case_Based': 2,
                'Genetic_Algorithms': 3,
                'Theory': 4,
                'Reinforcement_Learning': 5,
                'Probabilistic_Methods': 6
            },
            'citeseer': {
                'Agents': 0,
                'ML': 1,
                'IR': 2,
                'DB': 3,
                'HCI': 4,
                'AI': 5
            },
            'pubmed': {
                'Diabetes Mellitus, Experimental': 0,
                'Diabetes Mellitus Type 1': 1,
                'Diabetes Mellitus Type 2': 2
            },
            'wikics': {
                'Computational linguistics': 0,
                'Databases': 1,
                'Operating systems': 2,
                'Computer security': 4,
                'Computer architecture': 3,
                'Internet protocols': 5,
                'Computer file systems': 6,
                'Distributed computing architecture': 7,
                'Web technology': 8,
                'Programming language topics': 9
            }

            }

        # ID classes indices mapping
        id_classes_indices_mapping = {
            'cora': [4, 2, 5, 6],
            'citeseer': [2,3],
            'pubmed': [0, 1],
            'wikics': [1,4,5,6],

        }

        # OOD classes indices mapping
        ood_classes_indices_mapping = {
            'cora': [0, 1, 3],
            'citeseer': [0,1,4,5],
            'pubmed': [2],
            'wikics': [0,2,3,7,8,9],
        }

        
        def invert_mapping(mapping):
            """
            Inverts a dictionary mapping class names to indices to a dictionary mapping indices to class names.
            
            Args:
                mapping (dict): Original mapping from class names to indices.
                
            Returns:
                dict: Inverted mapping from indices to class names.
            """
            return {v: k for k, v in mapping.items()}

        def map_indices_to_labels(dataname, indices, full_mapping):
            """
            Maps a list of indices to their corresponding label names for a given dataset.
            
            Args:
                dataname (str): The name of the dataset.
                indices (list of int): List of class indices.
                full_mapping (dict): Mapping from dataset to label names.
                
            Returns:
                list of str: Corresponding label names in lowercase.
            """
            # Invert the mapping to map indices to labels
            inverted_mapping = invert_mapping(full_mapping[dataname])
            
            # Determine the maximum index to ensure the list is complete
            max_index = max(inverted_mapping.keys())
            
            # Create a list ordered by index
            label_names_ordered = [inverted_mapping.get(x, f"unknown_{x}") for x in range(max_index + 1)]
            
            return [label_names_ordered[x] for x in indices]


        inverted_mapping = invert_mapping(full_mapping[dataname])
        
        # Create an ordered list of label names based on indices
        max_index = max(inverted_mapping.keys())
    

        label_names_ordered = [inverted_mapping.get(x, f"unknown_{x}") for x in range(max_index + 1)]
        
        # Assign the ordered label names to data.label_names
        data.label_names = [label_names_ordered[x] for x in range(len(label_names_ordered))]
        data.label_names = [x.lower() for x in data.label_names]
        
        # Map indices to class names for ID and OOD
        id_class_indices = id_classes_indices_mapping.get(dataname, [])
        ood_class_indices = ood_classes_indices_mapping.get(dataname, [])

        id_classes = map_indices_to_labels(dataname, id_class_indices, full_mapping)
        ood_classes = map_indices_to_labels(dataname, ood_class_indices, full_mapping)
        
        

        exp = Experiment(data, key, data_path)
        object_cat = configs[dataname]['zero-shot']['object-cat']
        question = configs[dataname]['zero-shot']['question']
        answer_format = configs[dataname]['zero-shot']['answer-format']
        examples = configs[dataname]['few-shot']['examples']
        example_text = examples[0][0]
        example_labels = examples[0][1]
        few_shot_topk = configs[dataname]['few-shot-2']['examples']
        fst_example = few_shot_topk[0][0]
        fst_result = few_shot_topk[0][1]
        idxs = torch.arange(data.x.shape[0])
        performance = {}
        performance[dataname] = {}
        performance[dataname]['zero_shot'] = []
        performance[dataname]['few_shot'] = []
        performance[dataname]['cot'] = []
        performance[dataname]['topk'] = []
        performance[dataname]['consistency'] = []
        performance[dataname]['consistency_no_topk'] = []
        performance[dataname]['few_shot_all'] = []

        cost = {}
        cost[dataname] = {}
        cost[dataname]['zero_shot'] = []
        cost[dataname]['few_shot'] = []
        cost[dataname]['cot'] = []
        cost[dataname]['topk'] = []
        cost[dataname]['consistency'] = []
        cost[dataname]['consistency_no_topk'] = []
        cost[dataname]['few_shot_all'] = []

        for seed in seeds:
            select_samples = data.train_masks[seed]
            
            
            # Set the random seed for reproducibility
            torch.manual_seed(seed)

            select_ids = idxs[select_samples]
            if len(select_ids) > 300:
                select_ids = select_ids[torch.randperm(len(select_ids))[:20]]
            orig_question = data.raw_texts
            if dataname == '20newsgroup':
                orig_question = [clean_header(x) for x in tqdm(orig_question)]
                orig_question = [clean_text(x) for x in tqdm(orig_question)]
            if dataname == 'wikics' or dataname == 'products' or dataname == '20newsgroup':
                orig_question = [keep_first_n_words(x, 256) for x in tqdm(orig_question)]
            print("Label Names:", data.label_names)

            # Generate prompts using only ID classes
            questions = zero_shot_prompt(
                texts=orig_question,
                label_names=id_classes,  # Only ID classes
                need_tasks=True,
                object_cat=object_cat,

            )
            
            
            no_task_question = zero_shot_prompt(
                texts=orig_question,
                label_names=id_classes,  # Only ID classes
                need_tasks=False,
                object_cat=object_cat,

            )
        
            if 'draw' in exps:
                cbar = False
            # do plot here
                saved_results = exp.load_saved(dataname, 'few_shot_all')
                pred = saved_results['pred']
                conf = saved_results['conf']

                total_idxs = torch.arange(len(pred))
                random_idx = total_idxs[(pred != -1)][:1000]

                pred = pred[random_idx]
                gt = data.y[random_idx]


                axis_labels = [f"{i}" for i in range(len(data.label_names))]

                noise_transition_matrix(pred, gt, "{}_noise_transition_matrix_few_shot.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)


                xticks = [f"c{i}" for i in range(len(data.label_names))]
                count_and_plot(pred, gt, xtick_labels = xticks, output_name = "{}_few_shot_pred_label_distribution.pdf".format(dataname), title = "{} sampled llm label distribution".format(dataname))

                zero_shot_results = exp.load_saved(dataname, "consistency")
                pred = zero_shot_results['pred']
                conf = zero_shot_results['conf']

                total_idxs = torch.arange(len(pred))
                random_idx = total_idxs[(pred != -1)][:1000]

                pred = pred[random_idx]
                gt = data.y[random_idx]

                acc = (pred == gt).float().mean()

                noise_transition_matrix(pred, gt, "{}_noise_transition_matrix_zero_shot.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)


                count_and_plot(pred, gt, xtick_labels = xticks, output_name = "{}_zero_shot_pred_label_distribution.pdf".format(dataname), title = "{} sampled llm label distribution".format(dataname))

                new_y = inject_random_noise_y_level(gt, 1 - acc)

                noise_transition_matrix(new_y, gt, "{}_noise_transition_matrix_synthetic.pdf".format(dataname), x_axis_labels = axis_labels, y_axis_labels = axis_labels, cbar = cbar)
                


            if 'zero_shot' in exps:
                print("zero shot all")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                vanilla_response, error_idx = exp.async_api(vanilla_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'zero_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                exp.save_cache(dataname, 'zero_shot', vanilla_response)
                res = exp.retrieve_answer(vanilla_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='zero_shot', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['zero_shot'].append(eval_out)
                response_str = [x[0] for x in vanilla_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['zero_shot'].append(all_costs)
            
            if 'no_fix' in exps:
                print("No fix")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                vanilla_response, error_idx = exp.test_no_correction(vanilla_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'zero_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                res = exp.retrieve_answer(vanilla_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='no_fix', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['zero_shot'].append(eval_out)
                response_str = [x[0] for x in vanilla_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['zero_shot'].append(all_costs)

            
            if 'few_shot_all' in exps:
                print("few shot all")
                few_shot_all_prompts = few_shot_topk_prompts(orig_question, data.label_names, fst_example, fst_result, exp, object_cat, question, answer_format, name = dataname)
                prompt_cost = calculate_cost_from_a_list_of_texts(few_shot_all_prompts, select_ids)
                few_shot_all_prompts = pack_prompt(few_shot_all_prompts, select_ids, data.x.shape[0])
                few_shot_all_response, error_idx = exp.async_api(few_shot_all_prompts, data.label_names, 0.7, None, n = 3, dataset = dataname, key = 'few_shot_all', rewrite = False, fix = True, seed = seed, sp = 60, ss = 3)
                exp.save_cache(dataname, 'few_shot_all', few_shot_all_response)
                res = exp.retrieve_multiple_answers(few_shot_all_response, data, 3)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='few_shot_all', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['few_shot_all'].append(eval_out)
                response_str = [x[0] + x[1] + x[2] for x in few_shot_all_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['few_shot_all'].append(all_costs)

            ## few-shot vanilla performance
            if 'few_shot' in exps:
                print("few shot")
                few_shot_prompt = few_shot_prompts(orig_question, data.label_names, example_text, example_labels, object_cat, question, answer_format)
                prompt_cost = calculate_cost_from_a_list_of_texts(few_shot_prompt, select_ids)
                few_shot_prompt = pack_prompt(few_shot_prompt, select_ids, data.x.shape[0])
                few_shot_response, error_idx = exp.async_api(few_shot_prompt, data.label_names, temperature = 0, top_p = None, n = 1, dataset = dataname, key = 'few_shot', rewrite = False, fix = True, seed = seed, sp = 60, ss = 2)
                exp.save_cache(dataname, 'few_shot', few_shot_response)
                res = exp.retrieve_answer(few_shot_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='few_shot', strategy = args.strategy, data_obj = data, seed = seed, g_error_idx = error_idx)
                performance[dataname]['few_shot'].append(eval_out)
                response_str = [x[0] for x in few_shot_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['few_shot'].append(all_costs)



            ## topk performance
            if 'topk' in exps:
                print("topk")
                topk_step_prompt = [exp.topk_confidence(q, 3, dataname) for q in no_task_question]
                prompt_cost = calculate_cost_from_a_list_of_texts(topk_step_prompt, select_ids)
                topk_step_prompt = pack_prompt(topk_step_prompt, select_ids, data.x.shape[0])
                topk_response, error_idx = exp.async_api(topk_step_prompt, data.label_names, 0, None, n = 1, dataset = dataname, key = 'topk', rewrite = False, fix = True, seed = seed, sp = 60, ss = 0)
                exp.save_cache(dataname, 'topk', topk_response)
                res = exp.retrieve_answer(topk_response, data)
                eval_out, _, _ = exp.eval_result(res, data.label_names, select_ids, data.y, dataname, method='topk', strategy = args.strategy, data_obj = data, seed = seed)
                performance[dataname]['topk'].append(eval_out)
                response_str = [x[0] for x in topk_response if x != '']
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['topk'].append(all_costs)

            if 'consistency' in exps:
                print("consistency")
                topk_step_prompt = [exp.topk_confidence(q, 1, dataname) for q in no_task_question]
                topk_step_prompt = pack_prompt(topk_step_prompt, select_ids, data.x.shape[0])
                prompt_cost = calculate_cost_from_a_list_of_texts(topk_step_prompt, select_ids)
                acc, response_str = consistency(topk_step_prompt, dataname, exp, 0.7, data, strategy = args.strategy, seed = seed)

                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                performance[dataname]['consistency'].append(acc)
                cost[dataname]['consistency'].append(all_costs)

            if 'consistency_no_topk' in exps:
                print("consistency no topk")
                vanilla_prompt = [exp.vanilla_verbalized_confidence(q) for q in questions]
                vanilla_prompt = pack_prompt(vanilla_prompt, select_ids, data.x.shape[0])
                prompt_cost = calculate_cost_from_a_list_of_texts(vanilla_prompt, select_ids)
                acc, response_str = consistency_no_topk(vanilla_prompt, dataname, exp, 0, data, strategy = args.strategy, seed = seed, key_name = 'consistency_0_vanilla')

                performance[dataname]['consistency_no_topk'].append(acc)
                all_costs = prompt_cost + calculate_cost_from_a_list_of_texts(response_str)
                cost[dataname]['consistency_no_topk'].append(all_costs)
            
            
        if 'zero_shot' in exps:
            mean_test_acc = np.mean(performance[dataname]['zero_shot']) * 100
            std_test_acc = np.std(performance[dataname]['zero_shot']) * 100
            avg_cost = np.mean(cost[dataname]['zero_shot'])
            print(f"Zero-shot Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Zero-shot Average Cost: {avg_cost:.2f}")

        if 'few_shot' in exps:
            mean_test_acc = np.mean(performance[dataname]['few_shot']) * 100
            std_test_acc = np.std(performance[dataname]['few_shot']) * 100
            avg_cost = np.mean(cost[dataname]['few_shot'])
            print(f"Few-shot Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Few-shot Average Cost: {avg_cost:.2f}")

        if 'topk' in exps:
            mean_test_acc = np.mean(performance[dataname]['topk']) * 100
            std_test_acc = np.std(performance[dataname]['topk']) * 100
            avg_cost = np.mean(cost[dataname]['topk'])
            print(f"Top-k Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Top-k Average Cost: {avg_cost:.2f}")

        if 'consistency' in exps:
            mean_test_acc = np.mean(performance[dataname]['consistency']) * 100
            std_test_acc = np.std(performance[dataname]['consistency']) * 100
            avg_cost = np.mean(cost[dataname]['consistency'])
            print(f"Consistency Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Consistency Average Cost: {avg_cost:.2f}")

        if 'consistency_no_topk' in exps:
            mean_test_acc = np.mean(performance[dataname]['consistency_no_topk']) * 100
            std_test_acc = np.std(performance[dataname]['consistency_no_topk']) * 100
            avg_cost = np.mean(cost[dataname]['consistency_no_topk'])
            print(f"Consistency No Top-k Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Consistency No Top-k Average Cost: {avg_cost:.2f}")
        
        if 'few_shot_all' in exps:
            mean_test_acc = np.mean(performance[dataname]['few_shot_all']) * 100
            std_test_acc = np.std(performance[dataname]['few_shot_all']) * 100
            avg_cost = np.mean(cost[dataname]['few_shot_all'])
            print(f"Few-shot All Test Accuracy: {mean_test_acc:.2f} ± {std_test_acc:.2f}")
            print(f"Few-shot All Average Cost: {avg_cost:.2f}")









    



