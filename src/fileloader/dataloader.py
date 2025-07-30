import json
import os
import warnings
import re
import time
import pandas
import tqdm
from modelscope.msdatasets import MsDataset
from PIL import Image
import io

class qapair():
    def __init__(self,question,answer,image,id):
        self.question=question
        self.answer=answer
        self.image=image
        self.id=id

    def choice(self,choice):
        self.choice=choice



def format_image_name(image_id,split="val"):
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
    return f"COCO_{split}2014_{image_id:012d}.jpg"

class datas():
    def __init__(self,datapath,split="val"):
        self.datatype="okvqa"
        self.split=split
        print("initialing the datas")
        qp=os.path.join(datapath,f"OpenEnded_mscoco_{type}2014_questions.json")
        ap=os.path.join(datapath,f"mscoco_{type}2014_annotations.json")
        ip=os.path.join(datapath,f"{type}2014")
        self.image_path=ip
        self.question,self.answer=self.load_json(qp,ap,ip,split)
        self.processdatas(split=split)
        print("finish loading datas")

    def load_json(self,question_path, answer_path, image_path,split="val"):
        with open(question_path) as f:
            data = json.load(f)
            question = data["questions"]
            with open(answer_path) as an:
                data = json.load(an)
                answer = data["annotations"]
            return question, answer

    def processdatas(self,split="val"):
        """
        combined每一个数据的数据结构
        {'id': 297147,
        'question': 'What sport can you use this for?',
        'image_path': '../data/OKVQA/val2014/COCO_val2014_000000297147.jpg',
        'answer': [{'answer_id': 1, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 2, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 3, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 4, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 5, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 6, 'raw_answer': 'racing', 'answer_confidence': 'yes', 'answer': 'race'}, {'answer_id': 7, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 8, 'raw_answer': 'motocross', 'answer_confidence': 'yes', 'answer': 'motocross'}, {'answer_id': 9, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}, {'answer_id': 10, 'raw_answer': 'riding', 'answer_confidence': 'yes', 'answer': 'ride'}]}
        """
        self.combined=[]
        i=0
        for each in self.question:
            id=each["question_id"]
            for ans in self.answer:
                if ans["question_id"]==id:
                    temp=qapair(each["question"],ans["answers"],os.path.join(self.image_path,format_image_name(each["image_id"],split=self.split)),each["question_id"])
                    self.combined.append(temp)
                    break
        print(self.combined[0].id,self.combined[0].question,self.combined[0].answer,self.combined[0].image)


    def getanswer(self,id):
        for each in self.answer:
            if each.id == id:
                return each["answers"]
        warnings.warn("didn't find answer which match the id")
        return None

    def getquestion(self,id):
        for each in self.combined:
            if each.id == id:
                return each.question
        warnings.warn("didn't find question which match the id")
        return None

    def getimage(self,id):
        for each in self.combined:
            if each.id == id:
                return each.image
        warnings.warn("didn't find image which match the id")
        return None

    def evaluate_jsonl(self, jsonl_path):
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id=data.get("id")
                print(id)
                answers=self.getanswer(id)

                # 这里要获取模型输出的答案，字段名是 "answer"
                predicted_answer = data.get('answer', '')  # ← 修改这里
                if not isinstance(predicted_answer, str):
                    predicted_answer = str(predicted_answer)  # 确保是字符串

                # 处理预测的答案：去除所有空格并转为小写
                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())

                # 获取真实答案
                for each in answers:
                    ans = each['answer'].replace(" ", "")
                    raw_ans = each['raw_answer'].replace(" ", "")
                    if ans in processed_pred or raw_ans == processed_pred:
                        correct_count += 1
                        break
                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f"✅ 正确数: {correct_count} / 总数: {total_count}")
        print(f"🎯 准确率: {accuracy:.2f}%")

        return correct_count, total_count, accuracy




def format(image_id, split="val"):
        return f"abstract_v002_{split}2015_{image_id:012d}.png"


class datap():
    def __init__(self,datapath,split="val"):
        print("initializing data")
        ds=pandas.read_parquet(datapath)
        self.lenth=len(ds)
        print(self.lenth)
        print(type(ds))
        print(ds.iloc[0])
        self.combined=[]
        for i in range(self.lenth):
            imgpath=os.path.join("/root/autodl-tmp/RoG/qwen/data/AOKVQA/img",ds.iloc[i]["question_id"]+".png")
            if os.path.exists(imgpath):
                temp = qapair(ds.iloc[i]["question"], ds.iloc[i]["direct_answers"], imgpath, ds.iloc[i]["question_id"])
                temp.choice(ds.iloc[i]["choices"])
            else:
                print("find nothing")
                image_bytes = ds.iloc[i]["image"]['bytes']
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                temp=qapair(ds.iloc[i]["question"],ds.iloc[i]["direct_answers"],image,ds.iloc[i]["question_id"])
                temp.choice(ds.iloc[i]["choices"])
            self.combined.append(temp)
        print("loading finish")

    def createimg(self):
        for each in tqdm.tqdm(self.combined):
            imgpath=os.path.join("/root/autodl-tmp/RoG/qwen/data/AOKVQA/img",each.id+".png")
            each.image.save(imgpath)
            each.image=imgpath


    def getanswer(self,image_id):
        for each in self.combined:
            if each.id == image_id:
                return each.answer
        warnings.warn("didn't find answer whitch match the id")
        return None

    def evaluate_jsonl(self, jsonl_path):
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id=data.get("id")
                answers=self.getanswer(id)

                # 这里要获取模型输出的答案，字段名是 "answer"
                predicted_answer = data.get('predicted_answer', '')
                if not isinstance(predicted_answer, str):
                    predicted_answer = str(predicted_answer)  # 确保是字符串

                # 处理预测的答案：去除所有空格并转为小写
                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())
                answers=answers.replace("[","").replace("]","").split(",")


                for each in answers:
                    temp=each.replace("'", "").replace(" ", "")
                    print(temp)
                    if temp in processed_pred :
                        correct_count += 1
                        break
                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f"✅ 正确数: {correct_count} / 总数: {total_count}")
        print(f"🎯 准确率: {accuracy:.2f}%")

        return correct_count, total_count, accuracy

class datav():#给vqa用
    def __init__(self, datapath, split="val"):
        self.ans_path = f"abstract_v002_{split}2017_annotations.json"
        self.que_path = f"OpenEnded_abstract_v002_{split}2017_questions.json"
        self.image_path = os.path.join(datapath,f"scene_img_abstract_v002_{split}2017")
        inputanswer = os.path.join(datapath, self.ans_path)
        inputquestion = os.path.join(datapath, self.que_path)
        self.combined = []
        print("data initial")
        with open(inputanswer, "r") as f1:
            with open(inputquestion, "r") as f2:
                data = json.load(f1)
                que = json.load(f2)
                answers = data["annotations"]
                question = que["questions"]
                print(len(question))
                print(len(answers))
                time1 = time.time()
                for que in question:
                    question_id = que["question_id"]
                    for ans in answers:
                        if ans["question_id"] == question_id:
                            temp = qapair(que["question"], ans["answers"],
                                          os.path.join(self.image_path, format(ans["image_id"])), question_id)
                            self.combined.append(temp)
                            break
                        # print("没找到")
                print(f"dataload finish,cost {time.time() - time1}s")
                print(len(self.combined))
                # for each in self.combined:
                #     print(each.answer[0]["answer"])


class dataf():
    def __init__(self, qapath,imagepath, splitof="val"):
        with open(qapath, "r") as f1:
            self.combined = []
            data = json.load(f1)
            print(type(data))
            for each in data:
                temp=qapair(
                        data[each]["question"],
                    data[each]["answer"],
                    os.path.join(imagepath,data[each]["img_file"]),
                    each
                    )
                self.combined.append(temp)
        self.length=len(self.combined)
        self.num_train=self.length*4/5
        self.num_val=self.length-self.num_train
        self.train=[]
        self.val=[]
        for i in range(self.length):
            if i< self.num_train:
                self.train.append(self.combined[i])
            else:
                self.val.append(self.combined[i])

    def getquestion(self,id):
        for each in self.combined:
            if each.id == id:
                return each.question
            else:
                return None

    def getanswer(self,image_id):
        for each in self.combined:
            if each.id == image_id:
                return each.answer
        warnings.warn("didn't find answer whitch match the id")
        return None

    def evaluate_jsonl(self, jsonl_path):
        correct_count = 0
        total_count = 0

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                id=data.get("id")
                answers=self.getanswer(id)

                predicted_answer = data.get('predicted_answer', '')
                if not isinstance(predicted_answer, str):
                    predicted_answer = str(predicted_answer)  # 确保是字符串

                processed_pred = re.sub(r'\s+', '', predicted_answer.lower())
                answers=answers.replace("[","").replace("]","").split(",")

                flag=1
                temp=answers[0]
                temp=temp.replace("'", "").replace("a ", "").replace(" ", "")
                if temp.endswith("s"):
                    temp=temp[:-1]
                temp=temp.lower()
                if temp in processed_pred :
                    correct_count += 1
                    flag=0
                if flag==1:
                    print([temp,processed_pred])

                total_count += 1

        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
        print(f"✅ 正确数: {correct_count} / 总数: {total_count}")
        print(f"🎯 准确率: {accuracy:.2f}%")

        return correct_count, total_count, accuracy


if __name__ == "__main__":
    # qapath="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/all_qs_dict_release.json"
    # image="/root/autodl-tmp/RoG/qwen/data/FVQA/new_dataset_release/images"
    # ds=dataf(qapath,image)
    # for each in ds.train:
    #     print(each.id)
    # ds=datap("/root/autodl-tmp/RoG/qwen/data/AOKVQA/data/test-00000-of-00001-d306bf3ad53b6618.parquet")
    # ds.createimg()
    ds=datas("/root/autodl-tmp/RoG/qwen/data/OKVQA")
