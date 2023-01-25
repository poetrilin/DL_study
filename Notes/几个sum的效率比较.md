参考博客[(22条消息) Python内置函数与numPy运算速度对比_Yeuing的博客-CSDN博客](https://blog.csdn.net/Yeuing/article/details/38018183)

Python自己带了几个函数，主要是sum,max,min，同时numPy中也有几个类似的函数，今天对比了一下几个函数的运算速度，发现了还是numpy的array计算速度最快。

思路，通过产生1万个随机数，对其用四种方法求和，以及求最大值，求均值的方式与求和相同，求最小值的方式与求最大值也类似，故只测了求和与最大值两项。

```python
import random
import time
import numpy as np
from pandas import Series
a=[]
for i in range(100000000):
    a.append(random.random())
t1=time.time()
sum1=sum(a) #直接用内置函数求
t2=time.time()
sum2=np.sum(a)#用numpy直接求
t3=time.time()

b=np.array(a)
t4=time.time()
sum3=np.sum(b)#用numpy转换为array后求
t5=time.time()

c=Series(a)
t6=time.time()
sum4=c.sum()#用pandas的Series对象求
t7=time.time()
print t2-t1,t3-t2,t5-t4,t7-t6
```
最后的结果分别为# sum 1.60611581802       9.87746500969    0.223296165466   1.66015696526
可以看出，以array为对象的numpy计算方式最快，而以numpy直接计算最慢，内置函数速度排第二。

```python
求最大值

<pre name="code" class="python">import random
import time
import numpy as np
from pandas import Series
a=[]
for i in range(100000000):
    a.append(random.random())
t1=time.time()
sum1=max(a)#直接用内置函数求
t2=time.time()
sum2=np.max(a)#用numpy直接求
t3=time.time()
b=np.array(a)
t4=time.time()
sum3=np.max(b)#用numpy转换为array后求
t5=time.time()
c=Series(a)
t6=time.time()
sum4=c.max()#用pandas的Series对象求
t7=time.time()
print t2-t1,t3-t2,t5-t4,t7-t6
```

结果为：

max 2.81509399414    9.83987283707   0.219717025757    1.62969207764

结果依然是以array为计算对象的numpy最快。

综上，如果考虑运算速度，应该先将对象转为array，再用numpy进行计算，可获得最快的计算速度。