---
layout: page
title: "project"
header-img: "img/zhihu.jpg"
description: "project"
---

#### NER
2018.11  
[**[dataset](https://github.com/kifish/NER-demo/tree/master/data)**]  [**[code](https://github.com/kifish/NER-demo)**]  
尝试了5种方法实现中文命名实体词识别:    
1.[HMM](https://github.com/kifish/NER-demo/tree/hmm)  
2.[CRF](https://github.com/kifish/NER-demo/tree/crf)  
3.[BiLSTM-viterbi](https://github.com/kifish/NER-demo/tree/BiLSTM-viterbi)  
4.[BiLSTM-CRF](https://github.com/kifish/NER-demo/tree/BiLSTM-crf)  
5.[BiLSTM-CNN-CRF](https://github.com/kifish/NER-demo/tree/BiLSTM-cnn-crf)

#### semantic parsing
2019.05  
natural language -> logical form  
[**[dataset](https://github.com/msra-nlc/MSParS)**]  [**[code](https://github.com/kifish/ml-base/tree/master/pku-deep-learning/wxj-course/%E8%AF%AD%E4%B9%89%E8%AE%A1%E7%AE%97%E4%B8%8E%E7%9F%A5%E8%AF%86%E6%A3%80%E7%B4%A2/project/src)**]

method:  
bilstm-crf ner + seq2seq     
TODO: maybe pointer network + seq2seq is better!

#### 基于知识选择的检索式chatbot
2020.02-2020.05   
在严睿老师组里实习。  
目前,在Persona-Chat和CMUDoG数据集上已有sota结果。  
To be released.

#### 实习经历
    
##### 医疗文本结构化
2018.05-2018.08                         志诺维思  
担任自然语言处理实习生，负责肿瘤病理数据结构化的工作，通过正则及医学规则实现了病理信息抽取，结构化结果由人工评估，准确率高达99%。并做了一些数据分析工作，使用[apriori](https://kifish.github.io/2018/07/24/apriori/)算法挖掘免疫组化抗体之间的[关联性](https://kifish.github.io/2018/07/24/apriori/),并实现了**[可视化](https://kifish.github.io/R-notes/plot_rules/qfs.html)**。 

##### 篇章理解
2019.11-2020.02                        字节跳动-搜索  
担任搜索算法实习生，负责网页信息提取，通过模板引擎及模板配置实现网页信息结构化, 实现了一个json diff组件；负责网页内容的**篇章理解**，通过根据步骤词建树以及子标题检测实现；负责精准问答，通过规则实现短答案区间检测，类似于Watson DeepQA(不完全相同)。

