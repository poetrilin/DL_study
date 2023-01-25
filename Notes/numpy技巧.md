# Numpy tricks in DL
一. ramdon库
1.numpy.random.rand()
用法是：numpy.random.rand(d0,d1,…dn)
以给定的形状创建一个数组，并在数组中加入在[0,1]之间均匀分布的随机样本。
用法及实现：

```python
 np.random.rand(3,2)
array([[ 0.14022471,  0.96360618],  #random
       [ 0.37601032,  0.25528411],  #random
       [ 0.49313049,  0.94909878]]) #random

np.random.rand(5)
array([ 0.26677034,  0.01680242,  0.5164905 ,  0.70920141,  0.30438513])
 ```
2.numpy.random.randn()
用法是：numpy.random.rand(d0,d1,…dn)
以给定的形状创建一个数组，数组元素来符合标准正态分布N(0,1)
若要获得一般正态分布这里写图片描述则可用sigma * np.random.randn(…) + mu进行表示
用法及实现：  
```py
 a = np.random.randn(2, 4)
array([[-0.29188711,  0.76417681,  1.00922644,  0.34169581],
       [-0.3652463 , -0.9158214 ,  0.34467129, -0.31121017]])
 b = np.random.randn(2)
array([ 0.37849173,  1.14298464])
```

3.numpy.random.randint()
用法是：numpy.random.randint(low,high=None,size=None,dtype)
生成在半开半闭区间[low,high)上离散均匀分布的整数值;若high=None，则取值区间变为[0,low)
用法及实现
high=None的情形
```py
>>> a = np.random.randint(2, size=10)
>>> a
array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1])
>>> b = np.random.randint(1, size=10)
>>> b
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
>>> c =  np.random.randint(5, size=(2, 4))
>>> c
array([[3, 4, 3, 3],
       [3, 0, 0, 1]])
high≠None
 
d = np.random.randint(2,high=6,size=(2,4))
>>> d
array([[5, 2, 4, 2],
       [4, 3, 5, 4]])
```
4.numpy.random.random_integers()
用法是： numpy.random.random_integers(low,high=None,size=None)
生成闭区间[low,high]上离散均匀分布的整数值;若high=None，则取值区间变为[1,low]
用法及实现
high=None的情形

```py
 np.random.random_integers(1, 6, 10)
array([4, 5, 2, 3, 4, 2, 5, 4, 5, 4])
>>> np.random.random_integers(6)
5<br>>>> np.random.random_integers(6,size=(3,2))<br>array([[1, 3],<br>       [5, 6],<br>       [3, 4]])
 ```

high≠None的情形
```py
 c =  np.random.random_integers(6,high=8,size=(3,2))
>>> c
array([[7, 8],
       [7, 8],
       [8, 8]])
```
此外，若要将【a,b】区间分成N等分，也可以用此函数实现
`a+(b-a)*(numpy.random.random_integers(N)-1)/(N-1)`

5.numpy.random_sanmple()
用法是： numpy.random.random_sample(size=None)
以给定形状返回[0,1)之间的随机浮点数
用法及实现
```py
>>> np.random.random_sample()
0.2982524530687424
>>> np.random.random_sample((5,))
array([ 0.47989216,  0.12580015,  0.99624494,  0.14867684,  0.56981553])
>>> np.random.random_sample((2,5))
array([[ 0.00659559,  0.45824325,  0.13738623,  0.60766919,  0.39234638],
       [ 0.6914948 ,  0.92461145,  0.43289058,  0.63093292,  0.06921928]])
```
其他函数，numpy.random.random() ;numpy.random.ranf()
numpy.random.sample()用法及实现都与它相同

6.numpy.random.choice()
用法是： `numpy.random.choice(a,size=None,replace=True,p=None)`
若a为数组，则从a中选取元素；若a为单个int类型数，则选取range(a)中的数
replace是bool类型，为True，则选取的元素会出现重复；反之不会出现重复
p为数组，里面存放选到每个数的可能性，即概率
 用法及实现
```py
a =  np.random.choice(5, 3)
>>> a
array([4, 3, 1])
>>>b =  np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
>>> b
array([2, 3, 3], dtype=int64)
>>> c =  np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
>>> c
array([3, 2, 0])