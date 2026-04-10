---
layout: page
title: "About"
description: "kifish"
header-img: "img/zhihu.jpg"
---

[Google Scholar](https://scholar.google.com/citations?user=bTRtpnIAAAAJ) [Github](https://github.com/kifish) Email: kifish.pro@gmail.com

Experience<br>
- 2026.04-present solo at K.ai<br>
- 2023.04-2026.04 LLM Researcher at ByteDance Seed LLM<br>
- 2021.07-2023.03 NLP algorithm engineer at Kuaishou MMU

Publications

2026
- [LLM pretrain] Understanding by Reconstruction: Reversing the Software Development Process for LLM Pretraining. Zhiyuan Zeng, Yichi Zhang, Yong Shan, **Kai Hua**, Siyuan Fang, Zhaiyu Liu, Jiaheng Liu, Haozhe Wang, Yining Zheng, Ming Ding, Ke Shen, Ge Zhang, Wenhao Huang, Xipeng Qiu. arXiv:2603.11103, 2026.03<br>
  - Great Team Collaboration
  - We propose Understanding by Reconstruction (UbR), a pretraining paradigm that reconstructs the latent development trajectories behind software repositories, exposing LLMs to the planning, reasoning, debugging, and iterative refinement processes hidden in static code.
  - I provide advice on data synthesis, design the data schema for continuous pretraining (CT) that organizes reconstructed development trajectories into a scalable training format, and conduct the experiments and empirical analysis.
  - [arXiv](https://arxiv.org/abs/2603.11103) [Twitter](https://x.com/zhiyuan_nlper/status/2033782539358445759) [Hugging Face](https://huggingface.co/papers/2603.11103)

2025
- [LLM pretrain] Xingwei Qu, Shaowen Wang, Zihao Huang, **Kai Hua**, Fan Yin, Rui-Jie Zhu, Jundong Zhou, Qiyang Min, Zihao Wang, Yizhi Li, Tianyu Zhang, He Xing, Zheng Zhang, Yuxuan Song, Tianyu Zheng, Zhiyuan Zeng, Chenghua Lin, Ge Zhang, Wenhao Huang. Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space. arXiv:2512.24617, 2025.10<br>
  - Great Team Collaboration
  - We propose **Dynamic Large Concept Models (DLCM)**, a hierarchical language modeling framework that learns semantic boundaries from latent representations and shifts computation from tokens to a compressed concept space where reasoning is more efficient.
  - I design and construct the training data entirely from open-source data.
  - [arXiv](https://arxiv.org/abs/2512.24617) [Hugging Face](https://huggingface.co/papers/2512.24617)

- [LLM pretrain] Rui-Jie Zhu, Zixuan Wang, **Kai Hua**, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, Qiyang Min, Hongzhi Huang, Xun Zhou, Wei Ye, Jiaheng Liu, Jian Yang, Yunfeng Shi, Chenghua Lin, Enduo Zhao, Tianle Cai, Ge Zhang, Wenhao Huang, Yoshua Bengio, Jason Eshraghian. Scaling Latent Reasoning via Looped Language Models. arXiv:2510.25741, 2025.10<br>
  - Great Team Collaboration
  - We scale up **Looped Language Models (LoopLM)** to 2.6 billion parameters and complete pretraining on 7.7 trillion open-source tokens following a multi-stage data recipe encompassing Pretraining, Continual Training (CT), Long-CT, and Mid-Training. The resulting model is on par with SOTA language models of 2–3× size. **We open source all the model weights and the data recipe**.
  - I design and curate all pretraining data mixtures utilizing open-source data and provide key insights throughout the pretraining process.
  - [Project Page](https://ouro-llm.github.io) [arXiv](https://arxiv.org/abs/2510.25741) [Twitter](https://x.com/RidgerZhu/status/1983732551404679632) [Hugging Face](https://huggingface.co/papers/2510.25741) [机器之心](https://mp.weixin.qq.com/s/cArf8L2lspzCpeW6Yzc3Fw)

- [LLM pretrain] **Kai Hua**, Steven Wu, Ge Zhang. AttentionInfluence: Adopting Attention Head Influence for Weak-to-Strong Pretraining Data Selection. arXiv:2505.07293, 2025.05<br>
  - LLM Pretrain-data Selection (Idea Originator && Project Leader)
  - We propose AttentionInfluence, a training-free and supervision-free method for reasoning-centric data selection. By masking attention heads in a small pretrained model and measuring loss differences, we identify reasoning-intensive data that significantly improves the performance of larger models. Applied to a 7B model, our approach yields consistent gains on benchmarks like MMLU, GSM8K, and HumanEval—demonstrating an effective weak-to-strong scaling path for reasoning-focused pretraining.
  - [arXiv](https://arxiv.org/abs/2505.07293) [Twitter](https://x.com/GeZhang86038849/status/1922182593791066351) [量子位](https://mp.weixin.qq.com/s/FlP_m6WuWrvxrF4fvgyR9A) [Community Reproduction](https://github.com/alexfdom/attention-influence) [Submission Log](https://github.com/kifish/attention_influence)

- [LLM posttrain] Jinrui Liu, Jeff Wu, Xuanguang Pan, Gavin Cheung, Shuai Ma, Chongyang Tao. AIR: Post-training Data Selection for Reasoning via Attention Head Influence. arXiv:2512.13279, 2025.12<br>
  - LLM Posttrain-data Selection (Idea Originator && Project Leader)
  - We propose AIR (Attention Influence for Reasoning), a train-free and unsupervised framework for post-training data selection. AIR measures the influence of attention heads to estimate the reasoning intensity of samples and intermediate steps, enabling more effective data filtering for multi-step reasoning tasks. Our results on Qwen2.5-32B using the s1 dataset demonstrate consistent improvements across diverse reasoning benchmarks while maintaining strong generalization.

- [LLM posttrain] Xuanguang Pan, Chongyang Tao, Jiayuan Bai, Jianling Gao, Zhengwei Tao, Xiansheng Zhou, Gavin Cheung, Shuai Ma. EvolSQL: Structure-Aware Evolution for Scalable Text-to-SQL Data Synthesis. arXiv:2601.04875, 2026.01<br>
  - Great Team Collaboration
  - We proposes a structure-aware framework for generating high-quality Text-to-SQL training data. Instead of relying on uncontrolled LLM generation, EvolSQL systematically increases SQL complexity through syntax-tree-based transformation operators, enabling scalable and diverse data synthesis. Experiments show that models trained on EvolSQL data achieve strong performance and generalization with significantly less (1/18) data, highlighting the importance of structure-aware data construction for semantic parsing.

- [Model] In-Place Test-Time Training. Guhao Feng, Shengjie Luo, **Kai Hua**, Ge Zhang, Wenhao Huang, Di He, Tianle Cai. ICLR 2026.
  - Great Team Collaboration
  - [ICLR 2026 Oral](https://openreview.net/forum?id=dTWfCLSoyl)
  - [arXiv](https://arxiv.org/abs/2604.06169) [GitHub](https://github.com/ByteDance-Seed/In-Place-TTT) [Hugging Face](https://huggingface.co/papers/2604.06169) [量子位](https://mp.weixin.qq.com/s?__biz=MzIzNjc1NzUzMw==&mid=2247882085&idx=2&sn=8c181403345d630adbc7a1f01756eeb0) [Twitter](https://x.com/tianle_cai/status/2041705054886097155)


- [LLM evaluation] NL2Repo-Bench: Towards Long-Horizon Repository Generation Evaluation of Coding Agents. arXiv:2512.12730, 2025.12<br>
  - Great Team Collaboration
  - discussion and cooperation
  - labeled examples [case](https://github.com/multimodal-art-projection/NL2RepoBench/blob/main/test_files/pysondb-v2/start.md)
  - [arXiv](https://arxiv.org/abs/2512.12730) [Hugging Face](https://huggingface.co/papers/2512.12730) [机器之心](https://mp.weixin.qq.com/s/-NyK458UGvs-LRZJ-9Jb8g) [Twitter](https://x.com/GeZhang86038849/status/2000781002554380298)


- Seed VLM&LLM Team. [Seed-2.0](https://seed.bytedance.com/zh/seed2), Technical Report, 2026.02<br>
  - VLM&LLM&Model (Team Collaboration)
  - Construct **all newly added** long-context(128K/512K) CT data and long-context evaluation
  - [paper](https://lf3-static.bytednsdoc.com/obj/eden-cn/lapzild-tss/ljhwZthlaukjlkulzlp/seed2/0214/Seed2.0%20Model%20Card.pdf)

- Seed VLM&LLM Team. [Seed-1.8](https://github.com/ByteDance-Seed/Seed-1.8), Technical Report, 2025.12<br>
  - VLM&LLM (Team Collaboration)
  - Construct **all newly added** long-context(128K/512K) CT data and long-context evaluation
  - [github](https://github.com/ByteDance-Seed/Seed-1.8)

- Seed Model&LLM&VLM Team. [Seed-VWN](https://huggingface.co/papers/2511.11238), Technical Report, 2025.11<br>
  - Model&LLM&VLM (Team Collaboration)
  - Construct **all newly added** long-context(128K/512K) CT data and long-context evaluation
  - [arXiv](https://arxiv.org/abs/2511.11238)

- Seed LLM Team. [Seed OSS 36B](https://huggingface.co/collections/ByteDance-Seed/seed-oss-68a609f4201e788db05b5dcd), Open Source Model, 2025.08<br>
  - LLM Code/Pretrain (Team Collaboration)
  - Led the text mid-training and long-context(128K/512K) CT
  - [Hugging Face](https://huggingface.co/collections/ByteDance-Seed/seed-oss-68a609f4201e788db05b5dcd) [量子位](https://mp.weixin.qq.com/s/cr8Q7jlHm-7sCcAcrvUcCg)
- Seed LLM&VLM Team. [Seed-1.6](https://seed.bytedance.com/en/seed1_6), Technical Blog, 2025.06<br>
  - LLM&VLM&Model Pretrain (Team Collaboration)
  - Led the multimodal long-context(128K/512K) CT
  - [Technical Blog](https://seed.bytedance.com/en/seed1_6) [机器之心](https://mp.weixin.qq.com/s/hgAxLm09l7bs7wOKloQKQg)
- Seed VLM&LLM Team. Seed1.5-VL Technical Report. arXiv:2505.07062, 2025.05.<br>
  - LLM&VLM Pretrain (Team Collaboration)
  - Led the text long-context(128K/512K) CT
  - [arXiv](https://arxiv.org/abs/2505.07062) [机器之心](https://mp.weixin.qq.com/s/GgJVkh8IorB6MvqlxESJLw)
- Seed LLM Team. Seed-Thinking-v1.5: Advancing Superb Reasoning Models with Reinforcement Learning. arXiv:2504.13914. 2025.04
  - LLM Pretrain (Team Collaboration)
  - Core contributor for pretraining data
  - [arXiv](https://arxiv.org/abs/2504.13914v1) [量子位](https://mp.weixin.qq.com/s/wfiPEXHayAmwJwrGTAjD2Q)

2024

- [Embedding] Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang, Zhen Li, **Kai Hua**, Zhengwei Tao, and Shuai Ma. Llms are also effective embedding models: An in-depth overview. arXiv preprint arXiv:2412.12591, 2024.12
- [arXiv](https://arxiv.org/abs/2412.12591) [TOIS 2025]

2020

- [Retrieval-Based Chatbot] **Kai Hua**, Zhiyuan Feng, Chongyang Tao, Rui Yan, Lu Zhang. Learning to Detect Relevant Contexts and Knowledge for Response Selection in Retrieval-based Dialogue Systems. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM 2020), 2020.10
- [arXiv](https://arxiv.org/abs/2509.22845) [CIKM 2020]

<!-- Busuanzi 页面访问统计 -->
<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>

<!-- 显示单篇文章阅读量 -->
<span id="busuanzi_container_page_pv">
  📈 Page Views: <span id="busuanzi_value_page_pv"></span>
</span>
