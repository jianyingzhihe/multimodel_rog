import json
import os
import warnings
import re

def format_image_name(image_id,type="val"):
    """
    根据提供的image_id生成对应的COCO图片名。

    参数:
    image_id (int): 图片的编号。

    返回:
    str: 格式化后的图片名称。
    # 示例用法
    image_id = 9
    formatted_name = format_image_name(image_id)
    print(formatted_name)  # 输出: test.jpg
    """
    # print(f"COCO_{type}2014_{image_id:012d}.jpg")
    return f"COCO_{type}2014_{image_id:012d}.jpg"

class datas():
    def __init__(self,datapath,type="val"):
        print("initialing the datas")
        qp=os.path.join(datapath,f"OpenEnded_mscoco_{type}2014_questions.json")
        ap=os.path.join(datapath,f"mscoco_{type}2014_annotations.json")
        ip=os.path.join(datapath,f"{type}2014")
        self.question,self.image,self.answer=self.load_json(qp,ap,ip,type)
        self.processdatas(type=type)
        print("finish loading datas")

    def load_json(self,question_path, answer_path, image_path,type="val"):
        with open(question_path) as f:
            data = json.load(f)
            """
            print(type(data)):        <class 'dict'>
            for each in data:
                print(each,data[each]):
            license  {'url': 'http://creativecommons.org/licenses/by/4.0/', 'name': 'Creative Commons Attribution 4.0 International License'}
            data_subtype train2014
            task_type  Open - Ended
            questions[{'image_id': 51606, 'question': 'What..........
            """
            question = data["questions"]
            """        
            print(type(question)):list
            print(len(question)):9009
            print(question[0]):{'image_id': 51606, 'question': 'What is the hairstyle of the blond called?', 'question_id': 516065}
            print(type(question[0])):dict
            """
            image = [os.listdir(image_path), image_path]

            with open(answer_path) as an:
                data = json.load(an)
                answer = data["annotations"]
                """
                print(type(answer),len(answer)):<class 'list'> 9009
                print(answer[0])
                {'image_id': 51606, 'answer_type': 'other', 'question_type': 'four', 'question_id': 516065, 'answers': [{'answer_id': 1, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 2, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 3, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 4, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 5, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 6, 'raw_answer': 'pony tail', 'answer_confidence': 'yes', 'answer': 'pony tail'}, {'answer_id': 7, 'raw_answer': 'braid', 'answer_confidence': 'yes', 'answer': 'braid'}, {'answer_id': 8, 'raw_answer': 'braid', 'answer_confidence': 'yes', 'answer': 'braid'}, {'answer_id': 9, 'raw_answer': 'ponytail', 'answer_confidence': 'yes', 'answer': 'ponytail'}, {'answer_id': 10, 'raw_answer': 'ponytail', 'answer_confidence': 'yes', 'answer': 'ponytail'}], 'confidence': 3}
                """
            return question, image, answer

    def processdatas(self,type="val"):
        """
        combined每一个数据的数据结构
        {'id': 297147,
        'question': 'What sport can you use this for?',
        'image_path': '../data/OKVQA/val2014/COCO_val2014_000000297147.jpg',
        'answer': [{'answer_id': 1, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 2, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 3, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 4, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 5, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 6, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 7, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 8, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 9, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}, {'answer_id': 10, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}]}
        """
        self.combined=[]
        for each in self.question:
            temp={}
            temp["id"]=each["image_id"]
            temp["question"]=each["question"]
            temp["image_path"]=(os.path.join(self.image[1],format_image_name(each["image_id"],type)))
            temp["answer"]=self.getanswer(each["image_id"])
            self.combined.append(temp)

    def getanswer(self,image_id):
        for each in self.answer:
            if each["image_id"] == image_id:
                return each["answers"]
        warnings.warn("didn't find answer whitch match the id")
        return None

    def getquestion(self,id):
        for each in self.combined:
            if each["id"] == id:
                return each["question"]
        warnings.warn("didn't find question whitch match the id")
        return None

    def evaluate_jsonl(self,jsonl_path):
        """
        加载 JSONL 文件并评估模型预测的准确性。

        参数:
            jsonl_path (str): JSONL 文件路径
            dataset: 包含真实答案的数据集对象，必须有 dataset.combined 属性，
                     每个元素应包含 'question' 和 'answer' 字段（列表）

        返回:
            correct_count (int): 正确预测的数量
            total_count (int): 总问题数量
            accuracy (float): 准确率
        """
        # 构建 question -> answer 的映射
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                predicted_answer = data.get('predicted_answer', '')+''.join(data.get('prediction', ''))+data.get('answer', '')
                # 处理预测的答案：去除所有空格并转为小写
                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())
                for each in self.combined[total_count]['answer']:
                    each['answer'] = each['answer'].replace(" ", "")
                    each['raw_answer'] = each['raw_answer'].replace(" ", "")
                    if each['answer'] in processed_pred:
                        correct_count += 1
                        break
                    if each['raw_answer'] == processed_pred:
                        correct_count += 1
                        break
                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f"✅ 正确数: {correct_count} / 总数: {total_count}")
        print(f"🎯 准确率: {accuracy:.2f}%")

        return correct_count, total_count, accuracy
