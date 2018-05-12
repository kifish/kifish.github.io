---
tags: [python]
---
10 Minutes to pandas： https://www.cnblogs.com/chaosimple/p/4153083.html

读取csv：https://zhuanlan.zhihu.com/p/26618330

删除某一列：

http://blog.sciencenet.cn/home.php?mod=space&uid=645086&do=blog&id=884388
```python
import numpy as np
import pandas as pd
df = pd.DataFrame({"a":np.random.randn(4),"b":np.linspace(1,4,4),"c":["hp","es",9,1]})
print(df)
print("del:")
df2 = df.drop('a',axis=1)
print(df2)
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
#path = 'C:\\Users\\Administrator\\Desktop\\P00000001-ALL.csv'
path = 'D:\\P00000001-ALL.csv'
file = open(path)
clean_data = pd.read_csv(file,index_col=0,parse_dates = True) # -1
#2
#clean_data = data.drop('cmte_id',axis = 1) #delete the index and data ;
#del data['cmte_id']
#print(data.columns)
head_data = clean_data.head(5)
print(head_data)
#print(data)
#print("h:")
print("123456th record:")
print(clean_data.ix[3,:])
#3
#df_new = df.set_index('new_index_col')
#df_candidates = clean_data.drop_duplicates('cand_id')
#print(df_candidates.ix[14],:)
#4
from collections import OrderedDict
can_dict = OrderedDict()
df_cand = clean_data[['cand_nm']]
print(df_cand)
for idx in df_cand.index:
    #print(type(df_cand.loc[idx].values[0][0]))  str
    if df_cand.loc[idx].values[0][0] not in can_dict:
        firstname_list = []
        for ch in df_cand.loc[idx].values[0][0]:  #single char
            if ch !=',':
                firstname_list.append(ch)
            else:
                break
        firstname = "".join(firstname_list)
        can_dict[df_cand.loc[idx].values[0][0]] = firstname
print(can_dict)
#5
#Q3
import tabula

```

读取pdf：https://zhuanlan.zhihu.com/p/20910680
 也可以用tabula,但是win10上报错了,可能是java版本问题
CalledProcessError: Command '['java', '-jar', 'D:\\WinPython-64bit-2.7.10.3\\python-2.7.10.amd64\\lib\\site-packages\\tabula\\tabula-1.0.1-jar-with-dependencies.jar', '--pages', '1', '--guess', 'C:\\Users\\Administrator\\Desktop\\\xe7\xa8\x8b\xe5\xba\x8f\\1119\\__MACOSX\\A4\\Pages_22_from_Interim_Report_WHARF.pdf']' returned non-zero exit status 1


https://github.com/chezou/tabula-py
