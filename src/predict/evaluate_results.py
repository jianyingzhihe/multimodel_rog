import argparse
import glob
import json
import re
import string
from sklearn.metrics import precision_score
import sys
import os

# 添加上一级目录到路径中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # remove <pad> token:
    s = re.sub(r"\b(<pad>)\b", " ", s)
    s = " ".join(s.split())
    return s


def match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)
    return s2 in s1


def eval_acc(prediction, answer):
    matched = 0.
    for a in answer:
        if match(prediction, a):
            matched += 1
    return matched / len(answer)


def eval_hit(prediction, answer):
    for a in answer:
        if match(prediction, a):
            return 1
    return 0


def eval_f1(prediction, answer):
    if len(prediction) == 0:
        return 0, 0, 0
    matched = 0
    prediction_str = ' '.join(prediction)
    for a in answer:
        if match(prediction_str, a):
            matched += 1
    precision = matched / len(prediction)
    recall = matched / len(answer)
    if precision + recall == 0:
        return 0, precision, recall
    else:
        return 2 * precision * recall / (precision + recall), precision, recall


def extract_topk_prediction(prediction, k=-1):
    results = {}
    for p in prediction:
        if p in results:
            results[p] += 1
        else:
            results[p] = 1
    if k > len(results) or k < 0:
        k = len(results)
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:k]]


def eval_result(predict_file, cal_f1=True, topk=-1):
    # predict_file = os.path.join(result_path, 'predictions.jsonl')
    eval_name = "detailed_eval_result_top_{topk}.jsonl" if topk > 0 else 'detailed_eval_result.jsonl'
    detailed_eval_file = predict_file.replace('predictions.jsonl', eval_name)
    # Load results
    acc_list = []
    hit_list = []
    f1_list = []
    precission_list = []
    recall_list = []
    with open(predict_file, 'r') as f, open(detailed_eval_file, 'w') as f2:
        for line in f:
            try:
                data = json.loads(line)
            except:
                print(line)
                continue
            id = data['id']
            prediction = data['prediction']
            answer = data['ground_truth']
            if cal_f1:
                if not isinstance(prediction, list):
                    prediction = prediction.split("\n")
                else:
                    prediction = extract_topk_prediction(prediction, topk)
                f1_score, precision_score, recall_score = eval_f1(prediction, answer)
                f1_list.append(f1_score)
                precission_list.append(precision_score)
                recall_list.append(recall_score)
                prediction_str = ' '.join(prediction)
                acc = eval_acc(prediction_str, answer)
                hit = eval_hit(prediction_str, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps(
                    {'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit, 'f1': f1_score,
                     'precission': precision_score, 'recall': recall_score}) + '\n')
            else:
                acc = eval_acc(prediction, answer)
                hit = eval_hit(prediction, answer)
                acc_list.append(acc)
                hit_list.append(hit)
                f2.write(json.dumps(
                    {'id': id, 'prediction': prediction, 'ground_truth': answer, 'acc': acc, 'hit': hit}) + '\n')

    if len(f1_list) > 0:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(
            sum(hit_list) * 100 / len(hit_list)) + " F1: " + str(
            sum(f1_list) * 100 / len(f1_list)) + " Precision: " + str(
            sum(precission_list) * 100 / len(precission_list)) + " Recall: " + str(
            sum(recall_list) * 100 / len(recall_list))
    else:
        result_str = "Accuracy: " + str(sum(acc_list) * 100 / len(acc_list)) + " Hit: " + str(
            sum(hit_list) * 100 / len(hit_list))
    print(result_str)
    result_name = "eval_result_top_{topk}.txt" if topk > 0 else 'eval_result.txt'
    eval_result_path = predict_file.replace('predictions.jsonl', result_name)
    with open(eval_result_path, 'w') as f:
        f.write(result_str)


from loader import *

import json

def gethit(respath, datapath, output_file="../results/unmatched_samples.json"):
    ans = datas(datapath)
    flag = 0
    cnt = 0
    unmatched_list = []  # 用于保存未命中的样本

    with open(respath, 'r', encoding='utf-8') as f:
        res = json.load(f)

    for each in res:
        cnt += 1
        qid = each['id']
        raw_pred = each['res'][0]

        # 预处理预测文本
        pred_processed = raw_pred.replace(" ", "").lower().strip().replace(",", "").replace("-", "")

        # 获取问题和真实答案
        question = ans.getquestion(qid)
        ground_answers = ans.getanswer(qid)
        answers = [ga['raw_answer'].lower().strip().replace(" ", "") for ga in ground_answers]

        matched = False
        for answer in answers:
            if answer in pred_processed:
                flag += 1
                matched = True
                break

        if not matched:
            # 如果没有命中，记录到列表中
            unmatched_entry = {
                "id": qid,
                "question": question,
                "ground_truth": answers,
                "prediction": raw_pred
            }
            unmatched_list.append(unmatched_entry)

# 打印命中统计
    print(f"Hit: {flag}/{cnt} = {flag / cnt:.2%}")

    # 写入未命中的结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(unmatched_list, f, indent=4, ensure_ascii=False)
    print(f"Unmatched samples saved to {output_file}")

if __name__ == "__main__":
    gethit("../results/output.json", "../data/OKVQA")