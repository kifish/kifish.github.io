---
layout: page
title: "project"
header-img: "img/zhihu.jpg"
description: "project"
---

#### 实习经历


#### 基于知识增强的生成式chatbot
2020.09-2020.11                         字节跳动-AI Lab  
研究对话生成, 利用知识增强来提升GPT2在对话生成的任务上的效果。(Work in progress)         


#### 基于知识选择的检索式chatbot
2020.02-present         
研究多轮检索式对话, 在Persona-Chat、CMUDoG和Wizard数据集上取得sota结果, 发表于CIKM2020:         
[Learning to Detect Relevant Contexts and Knowledge for Response Selection in Retrieval-based Dialogue Systems](https://dl.acm.org/doi/abs/10.1145/3340531.3411967)

CIKM2020 Full Paper Research Track 一作

<!-- (CIKM2020 Full Paper Research Track 共一一作)

-->

<!--  
[**[paper](https://github.com/kifish/knowbot)**]  [**[dataset](https://github.com/kifish/knowbot)**]  [**[code](https://github.com/kifish/knowbot)**] -->



##### FAQ
2020.04-2020.07                        腾讯微信-模式识别中心  
担任nlp算法实习生，参与 **[微信对话开放平台](https://openai.weixin.qq.com/)** 的**FAQ**项目, 即检索式 chatbot。   
负责 es 后台接口以及数据更新服务。    
负责对话反馈分类。   
通过预训练模型(GPT2/mass)，从标准问生成相似问。

<div style='display: none'>
https://www.cnblogs.com/yangzhou33/p/8438461.html


对话开发平台介绍:
https://mp.weixin.qq.com/s/mmBa_-Tw395RzV0CEnYZ0w

小微来云南啦！微信对话开放平台正式发布

https://mp.weixin.qq.com/s/mmBa_-Tw395RzV0CEnYZ0w

知识在检索式对话系统的应用

https://mp.weixin.qq.com/s/L4eXFSwYHg-Bq-iyMHSo2Q


EMNLP2018 | 一个承上启下的回答才是好的回答

https://mp.weixin.qq.com/s/bjRxJd0Syu7HGzjicXEbZg


你好，小微！微信智言“小微”智能对话系统亮相2019微信公开课PRO

https://mp.weixin.qq.com/s/wBYSgOFDnoMMcOpfjm8zMg



适应的对话生成模型——解决数据不足的秘密武器

https://mp.weixin.qq.com/s/EyxcUITOuplp1kQrD3a5hw


机器人健忘症的福音——对话系统上下文
http://mp.weixin.qq.com/s?__biz=MzI5MTU5OTM1NQ==&mid=2247484085&idx=1&sn=f71f5f97fb65008ab9abf1f5ec2bb5ba&chksm=ec0f6421db78ed37a165bbca219019805b93ac73950881c7b75b7ad7663383631bec56b7bbfb&mpshare=1&scene=24&srcid=0720DfcGZPq1OmxOZv5boz2S&sharer_sharetime=1595236506930&sharer_shareid=0c9ba40fe1895b3bda5b5da7468fef81#rd


微信攻AI，有软有硬
https://mp.weixin.qq.com/s?__biz=MzI5MTU5OTM1NQ==&mid=2247485180&idx=1&sn=2ac8723a962f280a592f02d37a3df889&chksm=ec0f6068db78e97efa1bd4068e291af142345cd85e027ecd00b0c7b43567569cc4e4f90456cb&mpshare=1&scene=24&srcid=072053RR5J8oNO2cRuV8gIhj&sharer_sharetime=1595236421986&sharer_shareid=0c9ba40fe1895b3bda5b5da7468fef81#rd

</div>



##### 篇章理解
2019.11-2020.02                        字节跳动-搜索  
担任搜索算法实习生，负责网页信息提取，通过模板引擎及模板配置实现网页信息结构化, 实现了一个json diff组件；负责网页内容的**篇章理解**，通过根据步骤词建树以及子标题检测实现；负责精准问答，通过规则实现短答案区间检测，类似于Watson DeepQA(不完全相同)。
    
##### 医疗文本结构化
2018.05-2018.08                         志诺维思  
担任自然语言处理实习生，负责肿瘤病理数据结构化的工作，通过正则及医学规则实现了病理信息抽取，结构化结果由人工评估，准确率高达99%。并做了一些数据分析工作，使用[apriori](https://kifish.github.io/2018/07/24/apriori/)算法挖掘免疫组化抗体之间的[关联性](https://kifish.github.io/2018/07/24/apriori/),并实现了**[可视化](https://kifish.github.io/R-notes/plot_rules/qfs.html)**。 



#### tiny projects

#### semantic parsing
2019.05  
natural language -> logical form  
[**[dataset](https://github.com/msra-nlc/MSParS)**]  [**[code](https://github.com/kifish/ml-base/tree/master/pku-deep-learning/wxj-course/%E8%AF%AD%E4%B9%89%E8%AE%A1%E7%AE%97%E4%B8%8E%E7%9F%A5%E8%AF%86%E6%A3%80%E7%B4%A2/project/src)**]

method:  
bilstm-crf + ner + seq2seq     

TODO: maybe pointer network + seq2seq is better!

#### NER
2018.11  
[**[dataset](https://github.com/kifish/NER-demo/tree/master/data)**]  [**[code](https://github.com/kifish/NER-demo)**]  
尝试了5种方法实现中文命名实体词识别:    
1.[HMM](https://github.com/kifish/NER-demo/tree/hmm)  
2.[CRF](https://github.com/kifish/NER-demo/tree/crf)  [learn CRF tips](https://kifish.github.io/2018/07/13/CRF/)  
3.[BiLSTM-viterbi](https://github.com/kifish/NER-demo/tree/BiLSTM-viterbi)  
4.[BiLSTM-CRF](https://github.com/kifish/NER-demo/tree/BiLSTM-crf)  
5.[BiLSTM-CNN-CRF](https://github.com/kifish/NER-demo/tree/BiLSTM-cnn-crf)

update: methods above are out of date.

增加了bert的支持。

6.[Bert](https://github.com/kifish/NER-demo/tree/bert)
