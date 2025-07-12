from fileloader.google import *
mo=googlemod("../multimodels/google/gemma")
res=mo.inf_with_score("what is it","../data/OKVQA/test.jpg")
print(res)