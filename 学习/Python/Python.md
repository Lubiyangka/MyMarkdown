# Python

## 一 数据类型

Python属于动态类型程序语言，数据使用之前无需声明；也是强类型语言只能接收有明确定义的操作

### 1.1 数据分类

| 数据结构   | 属性     |
| ---------- | -------- |
| str        | Text     |
| int        | Numeric  |
| float      | Numeric  |
| complex    | Numeric  |
| list       | Sequence |
| tuple      | Sequence |
| range      | Sequence |
| dict       | Mapping  |
| set        | Set      |
| frozenset  | Set      |
| bool       | Boolean  |
| bytes      | Boolean  |
| bytearray  | Boolean  |
| memoryview | Boolean  |

### 1.2 字符串

字符串是一种类数组的格式分配空间。

#### 1.2.1 字符串的定位

```python
hello="Hello world!"
print(hello[0])
print(hello[1])
print(hello[-1])
```

> 输出

```
H
e
!
```
> 对应表
>

```
┌───┬───┬───┬───┬───┬───┬───┐
| m | y | b | a | c | o | n |
└───┴───┴───┴───┴───┴───┴───┘
  0   1   2   3   4   5   6   7
 -7  -6  -5  -4  -3  -2  -1
```

#### 1.2.2语法格式

表示方式：

- 单引号`'abc'`

- 双引号`"abc"`

- 三个单引号'''or三个双引号"""

  可以允许多行字符串，包含空格和换行 

  ```python
      print("""7
  第一行 顶到头
      第二行     隔了好远呢
      
      第四行
      6""")
  ```

  > 输出

  ```
  7
  第一行 顶到头
      第二行     隔了好远呢
      
      第四行
      6
  ```

单双引号不可以混合使用，由于没有字符类型，可以使用`’  ‘` `“ ”`来表示字符串，或者字符

### 1.2数据类型

#### 1.2.1 int

表示整数，八进制表示`0o101`,十六进制表示`0x101`，其中第一个为0，第二个为字母且不区分大小写。

#### 1.2.2 float

表示浮点数，可以使用科学计数法`1.232132e+9`,`1.23E-9`字母之后为位数且不区分大小写。

#### 1.2.3 complex

表示数学上的复数，表示方法同数学表达式1+3J，同样J不区分大小写

#### 1.2.3 bool

包含两个常量`True`，`False`。可以结合运算符`and`，`or`，`not` 进行判断

```python
    print(True and False)
    print(False or True)
    print(not False)
```

> 输出

```
False
True
True
```

### 1.3 序列&集合类型

> 以下将要介绍的是四种容器类型的数据结构

#### 1.3.1 list

表示列表，有数据组成，有顺序，可改变，中括号表示，数据类型可以不相同

```python
    exampleList = [1, "fi", "er", 2, False, (1 + 8j)]
    print(exampleList)
```

> 输出

```
[1, 'fi', 'er', 2, False, (1+8j)]
```

#### 1.3.2 tuple

```python
    exampleTuple = (1, "tuple", True)
    print(exampleTuple)
```

> 输出

```
(1, 'tuple', True)
```

#### 1.3.3 set

```python
    exampleSet = {"set", 93, "jin", 3, False}
    print(exampleSet)
```

> 输出

```
{False, 3, 'jin', 'set', 93}
```

#### 1.3.4 dict

```python
    exampleDict = {"id": 1, 'name': "FanNance", "phone": "134251777"}
    print(exampleDict)
```

> 输出

```
{'id': 1, 'name': 'FanNance', 'phone': '134251777'}
```

| 容器类型     | list | tuple  | set  | dict |
| ------------ | ---- | ------ | ---- | ---- |
| 符号         | []   | ()     | {}   | {}   |
| 顺序         | 有   | 有     | 无   | 无   |
| 能否更改内容 | 可以 | 不可以 | 可以 | 可以 |

### 1.4 空值 `None`

## 二 函数

### 2.1 print()函数

#### 2.1.1 输入格式

- 双引号`""`单引号`''`都可以使用

```python
print("这是print()函数")
print('这还是print()函数')
```

#### 2.1.2 函数原型

`print(value, ..., sep = ' ', end = '\n', file = sys.stdout)`

- `value`是要输出的值，支持多个输入，中间使用 `,`间隔

- `sep`设置两个字符串之间的间隔，默认为`' '`
- `end`设置最后一个值所要加的字符串，默认为`'\n'`
- `file`设置输出设备的参数

### 2.2 type()函数

可以通过该函数获取某一变量当前的数据类型

```python
    print(type(100))
    print(type("bf"))
    print(type(False))
    print(type(1.2))
    print(type(2 + 8j))
    print(type('hhh'))
```

> 输出

```
<class 'int'>
<class 'str'>
<class 'bool'>
<class 'float'>
<class 'complex'>
<class 'str'>
```

### 2.3 input()函数

#### 2.3.1 函数原型

`input(prompt = None)`

#### 2.3.2 函数链接`eval()`

将`input`函数获取的字符串转换为数值

### 2.4 zip()函数

```python
a = [1, 2, 3]
b = [4, 5, 6]
zipped = zip(a, b)
d = zip(*zipped)
```

```python
print(list(d))
print(list(zipped))
```

输出

[(1, 2, 3), (4, 5, 6)]

[(1, 4), (2, 5), (3, 6)]

### 2.5 map函数

## 三 第三方包

- NumPy ：数据与数据运算，如矩阵运算，傅里叶变换，线性代数
- Matplotlib ：2D可视化工具，绘制条形图，直方图，散点图，立体图，饼图，频谱图，数学函数等图形
-  SciPy ：科学计算，优化与求解，稀疏矩阵，线性代数，插值，特殊函数，统计函数，积分，傅里叶变换，信号处理，图像处理
- pandas ：数据处理与分析
- Django Pyramid Web2py Flask ：Web框架，快速开发网站
- Kivy Flexx Pywin32 PyQt WxPython ：GUI框架
- BeautifulSoup ： HTML/XML解析器
- Pillow ：图形处理
- PyGame  :多媒体与游戏软件开发
- Requests ：访问网络资料
- Scrapy ：网络爬虫包，可用来数据挖掘与统计
- SciKit-Learn TensorFlow Keras：机器学习 

## 四 Matplotlib

# 数学建模

- [x] 线性规划
  - [x] 非线性规划
  - [x] 整数规划
  - [x] 0-1规划
- [x] 微分方程
  - [x] 最小二乘法：线性&非线性
- [x] 差分方程
- [x] 代数方程
- [ ] 离散模型

## 一 规划

### 1.1 线性规划

需要明确两部分内容：

- 目标函数(max,min)即最值问题
- 约束条件(s.t.)

然后转化为标准形式

$$
\begin{gathered}min\operatorname{c}^Tx\\s.t.\begin{cases}Ax\leq b\\Aeq*x=beq\\lb\leq x\leq ub&\end{cases}\end{gathered}
$$

#### 1.1.1scipy求解

核心函数源码`scipy.optimize.linprog(c,A_ub=None,b_ub=None,A_eq=None,b_eq=None,bounds=None,method='interior-point',callback=None,options=None,x0=None)`

| 参数     | 含义                                     | 数据类型                                       |
| -------- | ---------------------------------------- | ---------------------------------------------- |
| c        | 线性目标函数的系数                       | 一维数组                                       |
| A_ub     | 不等式约束矩阵                           | 二维数组                                       |
| b_ub     | 不等式约束向量                           | 一维数组                                       |
| A_eq     | 等式约束矩阵                             | 二维数组                                       |
| b_eq     | 等式约束向量                             | 一维数组1*                                     |
| bounds   | 定义决策变量x xx的最小值和最大值         | n维数组n*2                                     |
| method   | 算法                                     | ‘interior-point’, ‘revised simplex’, ‘simplex’ |
| callback | 调用回调函数                             |                                                |
| options  | 求解器选项字典                           |                                                |
| x0       | 猜测决策变量的值，将通过优化算法进行优化 | 一维数组                                       |

- 等待被调用的参数 ，如果提供了回调函数，则算法的每次迭代将至少调用一次。回调函数必须接受单个 scipy.optimize.OptimizeResult

- 当前仅由’ revised simplex’ 方法使用此参数，并且仅当 x0 表示基本可行的解决方案时才可以使用此参数

###### 模板

$$
\max z=2x_1+3x_2-5x_3 \\ s.t.\left\{\begin{matrix}x_1+x_2+x_3=7\\2x_1-5x_2+x_3\geq10\\x_1+3x_2+x_3\leq12\\x_1,x_2,x_3\geq0\end{matrix}\right.
$$



```python
from scipy import optimize
import numpy as np
minC = np.array([2, 3, -5])
maxC = np.array([-i for i in minC])
A = np.array([[-2, 5, -1], [1, 3, 1]])
B = np.array([-10, 12])
A1 = np.array([[1, 1, 1]])
B1 = np.array([[7]])
bounds = [[0, None], [0, None], [0, None]]
res = optimize.linprog(maxC, A, B, A1, B1, bounds=bounds)
print(res.fun)
```

#### 1.1.2pulp求解

简化版

```python
import pulp
# 目标函数系数
z = [2, 3, 1]
# 约束条件
a = [[1, 4, 2], [3, 2, 0]]
b = [8, 6]
m = pulp.LpProblem(sense=pulp.LpMinimize)
x = [pulp.LpVariable(f'x{i}', lowBound=0) for i in range(len(z))]
m += pulp.lpDot(z, x)
for i in range(len(a)):
    m += (pulp.lpDot(a[i], x) >= b[i])
m.solve()
print(f'优化结果:{pulp.value(m.objective)}')
print(f'参数结果:{[pulp.value(var) for var in x]}')
```

利用字典(dict)定义变量

```python
import pulp

Ingredients = ["CHICKEN", "BEEF", "MUTTON", "RICE", "WHEAT", "GEL"]
costs = {
    "CHICKEN": 0.013,
    "BEEF": 0.008,
    "MUTTON": 0.010,
    "RICE": 0.002,
    "WHEAT": 0.005,
    "GEL": 0.001
}
proteinPercent = {
    "CHICKEN": 0.100,
    "BEEF": 0.200,
    "MUTTON": 0.150,
    "RICE": 0.000,
    "WHEAT": 0.040,
    "GEL": 0.000,
}
fatPercent = {
    "CHICKEN": 0.080,
    "BEEF": 0.100,
    "MUTTON": 0.110,
    "RICE": 0.010,
    "WHEAT": 0.010,
    "GEL": 0.000,
}
fibrePercent = {
    "CHICKEN": 0.001,
    "BEEF": 0.005,
    "MUTTON": 0.003,
    "RICE": 0.100,
    "WHEAT": 0.150,
    "GEL": 0.000,
}
saltPercent = {
    "CHICKEN": 0.002,
    "BEEF": 0.005,
    "MUTTON": 0.007,
    "RICE": 0.002,
    "WHEAT": 0.008,
    "GEL": 0.000,
}
prob = pulp.LpProblem(name="CatFood", sense=pulp.LpMinimize)
ingredientVars = pulp.LpVariable.dicts("Ingr", Ingredients, 0)
prob += pulp.lpSum([costs[i]*ingredientVars[i] for i in Ingredients]), "cost"
prob += pulp.lpSum([ingredientVars[i] for i in Ingredients]) == 100, "PercentagesSum"
prob += pulp.lpSum([proteinPercent[i] * ingredientVars[i] for i in Ingredients]) >= 8.0, "ProteinRequirement"
prob += pulp.lpSum([fatPercent[i] * ingredientVars[i] for i in Ingredients]) >= 6.0, "FatRequirement"
prob += pulp.lpSum([fibrePercent[i] * ingredientVars[i] for i in Ingredients]) <= 2.0, "FibreRequirement"
prob += pulp.lpSum([saltPercent[i] * ingredientVars[i] for i in Ingredients]) <= 0.4, "SaltRequirement"
prob.solve()
print(f'Status:{pulp.LpStatus[prob.status]}')
print(f'优化结果:{pulp.value(prob.objective)}')
print(f'参数结果:{[pulp.value(var) for var in prob.variables()]}')
```

#### 1.1.3实际问题--交通问题

$$
\begin{gathered}min\sum_{i=1}^m\sum_{j=1}^nc_{ij}x_{ij}\\s.t.\begin{cases}\sum_{j=1}^nx_{ij}=a_i\\\sum_{i=1}^mx_{ij}=b_j\\x_{ij}\geq0&\end{cases}\end{gathered}
$$

```python
from pprint import pprint
import numpy as np
import pulp
def transportationProblem(costs, xMAx, yMax, isMax=True):
    row = len(costs)
    col = len(costs[0])
    if isMax is True:
        prob = pulp.LpProblem(name='Transports Problem', sense=pulp.LpMaximize)
    else:
        prob = pulp.LpProblem(name='Transports Problem', sense=pulp.LpMinimize)
    var = [[pulp.LpVariable(f'x{i}{j}', lowBound=0, cat=pulp.LpInteger) for j in range(col)] for i in range(row)]
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    prob += pulp.lpDot(flatten(var), costs.flatten())
    for i in range(row):
        prob += (pulp.lpSum(var[i]) <= xMAx[i])
    for j in range(col):
        prob += (pulp.lpSum([var[i][j] for i in range(row)]) <= yMax[j])
    prob.solve()
    return {
        'objective': pulp.value(prob.objective),
        'var': [[pulp.value(var[i][j]) for j in range(col)] for i in range(row)]
    }


if __name__ == '__main__':
    costs = np.array([[ 500,  550,  630, 1000, 800,  700],
                      [ 800,  700,  600,  950, 900,  930],
                      [1000,  960,  840,  650, 600,  700],
                      [1200, 1040,  980,  860, 880,  780]])
    maxPlant = [76, 88, 96, 40]
    maxCultivation = [42, 56, 44, 39, 60, 59]
    res = transportationProblem(costs, maxPlant, maxCultivation)
    print(f'最大值为{res["objective"]}')
    print('个变量取值')
    pprint(res['var'])
```

补充解

```python
from pprint import pprint

import numpy as np
import pulp

# 目标函数系数
z = np.array([[160, 130, 220, 170],
              [140, 130, 190, 150],
              [190, 200, 230, 0]])
row = len(z)
column = len(z[0])
# 约束条件系数
limit = np.array([[1, 1, 1, 1],
                  [1, 1, 1, 1],
                  [1, 1, 1, 0]])
a = [[30, 70, 10, 10],
     [80, 140, 30, 50]]
b = [50, 60, 50]
m = pulp.LpProblem(sense=pulp.LpMinimize)
x = [[pulp.LpVariable(f'x{i}{j}', lowBound=0) for j in range(column)] for i in range(row)]
m += pulp.lpDot(z, x)
for i in range(row):
    m += pulp.lpDot(limit[i], x[i]) == b[i]
for j in range(column):
    m += (pulp.lpDot((limit[i][j] for i in range(3)), (x[i][j] for i in range(3))) >= a[0][j])
    m += (pulp.lpDot((limit[i][j] for i in range(3)), (x[i][j] for i in range(3))) <= a[1][j])
m.solve()
print(f'优化结果:{pulp.value(m.objective)}')
pprint([[pulp.value(x[i][j]) for j in range(column)] for i in range(row)])

```

#### 1.1.4 非线性规划问题的转化


$$
\begin{aligned}\min\quad&\sum_{i=1}^{n}|x_{i}|\\\mathrm{s.~t.}\quad&\quad Ax\leq b\end{aligned}
$$
其中
$$
x=\begin{bmatrix}x_1&\cdots&x_n\end{bmatrix}^T
$$
令
$$
x_i=u_i-\nu_i\text{ , }\mid x_i\mid=u_i+\nu_i
$$
记
$$
u=\begin{bmatrix}u_1&\cdots&u_n\end{bmatrix}^T,\quad\nu=\begin{bmatrix}\nu_1&\cdots&\nu_n\end{bmatrix}^T
$$
转化为
$$
\begin{aligned}&\min\quad\sum_{i=1}^n(u_i+\nu_i)\\&\mathrm{s.~t.}\quad\begin{cases}A(u-\nu)\leq b\\u,\nu\geq0&\end{cases}\end{aligned}
$$


### 1.2 整数规划

#### 1.2.1 概述

基本与线性规划相同，增加部分变量为整数约束。基本框架是分支定界法，排除整数约束条件获得**松弛模型**使用线性规划解决。如果有变量不为整数，然后再松弛模型上添加整数约束条件`x<=floor(A)` or `x>= ceil(A)`该过程叫做**分支**。一直判断直到所有变量全为整数，停止分支操作，形成一颗树。**定界**是叶子节点产生之后，给问题定一个下界。如果目标函数值小于下界，停止分支。分支与定界同时产生。

```python
import math
import sys

from scipy.optimize import linprog


def integerPro(c, A, b, Aeq, beq, t=1.0e-12):
    res = linprog(c, A, b, Aeq, beq)
    if type(res.x) is float:
        bestX = [sys.maxsize] * len(c)
    else:
        bestX = res.x
    bestVal = sum([x * y for x, y in zip(c, bestX)]), "zip()函数"
    if all(((x - math.floor(x)) < t or (math.ceil(x) - x) < t) for x in bestX):
        return bestVal, bestX
    else:
        ind = [i for i, x in enumerate(bestX) if (x - math.floor(x)) > t and (math.ceil(x) - x) > t][0]
        newCon1 = [0] * len(A[0])
        newCon2 = [0] * len(A[0])
        newCon1[ind] = -1
        newCon2[ind] = -1
        newA1 = A.copy()
        newA2 = A.copy()
        newA1.append(newCon1)
        newA2.append(newCon2)
        newB1 = b.copy()
        newB2 = b.copy()
        newB1.append(-math.ceil(bestX[ind]))
        newB2.append(math.floor(bestX[ind]))
        r1 = integerPro(c, newA1, newB1, Aeq, beq)
        r2 = integerPro(c, newA2, newB2, Aeq, beq)
        if r1[0] < r2[0]:
            return r1
        else:
            return r2     
```

匈牙利法

```python
import numpy as np
from scipy.optimize import linear_sum_assignment

cost = np.array([[25, 29, 31, 42],
                 [39, 38, 26, 20],
                 [34, 27, 28, 40],
                 [24, 42, 36, 23]])
row_ind, col_ind = linear_sum_assignment(cost)
print(cost[row_ind][col_ind])
print(cost[row_ind][col_ind].sum())
```

### 1.3 非线性规划

```python
import numpy as np
from scipy.optimize import minimize


def fun(args):
    a, b, c, d = args
    v = lambda x: (a + x[0]) / (b + x[1]) - c * x[0] + d * x[2]
    return v


def con(args):
    x1min, x1max, x2min, x2max, x3min, x3max = args
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - x1min},
            {'type': 'ineq', 'fun': lambda x: -x[0] + x1max},
            {'type': 'ineq', 'fun': lambda x: x[1] - x2max},
            {'type': 'ineq', 'fun': lambda x: -x[1] + x2max},
            {'type': 'ineq', 'fun': lambda x: x[2] - x3min},
            {'type': 'ineq', 'fun': lambda x: -x[2] + x3max})
    return cons


if __name__ == '__main__':
    args = (2, 1, 3, 4)
    args1 = (0.1, 0.9, 0.1, 0.9, 0.1, 0.9)
    cons = con(args1)
    x0 = np.asarray((0.5, 0.5, 0.5))
    res = minimize(fun(args), x0, method='SLSQP', constraints=cons)
    print(res.fun)
    print(res.success)
    print(res.x)
```

### 1.4 动态规划

```python
def dynamic_p() -> list:
    # 物品栏
    items = [
        {"name": "水", "weight": 3, "value": 10},
        {"name": "书", "weight": 1, "value": 3},
        {"name": "食物", "weight": 2, "value": 9},
        {"name": "小刀", "weight": 3, "value": 4},
        {"name": "衣服", "weight": 2, "value": 5},
        {"name": "手机", "weight": 1, "value": 10}
    ]
    # 约束条件
    max_capacity = 6
    dp = [[0] * (max_capacity + 1) for _ in range(len(items) + 1)]
    for i in range(1, len(items)+1):
        for j in range(1, max_capacity+1):
            weight = items[i-1]["weight"]
            value = items[i-1]["value"]
            if weight>j:
                dp[i][j] = dp[i-1][j]
            else:
                dp[i][j] = max(value+dp[i-1][j-weight], dp[i-1][j])
    return dp


dp = dynamic_p()
for i in dp:
    print(i)
# 最优解
print(dp[-1][-1])
```

### 1.5多目标规划
为了求多目标规划的非劣解，需要将多目标转化成单目标。
- 效用最优化模型：对每组目标函数进行加权处理，整合成一组目标函数
- 理想点法（罚款模型）：
## 二 数值逼近

### 2.1 一维插值

[Python之建模数值逼近篇--一维插值_python 一维的邻近插值_](https://blog.csdn.net/weixin_45508265/article/details/113095126?ops_request_misc=&request_id=&biz_id=102&utm_term=一维插值python&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-113095126.142^v88^insert_down1,239^v2^insert_chatgpt&spm=1018.2226.3001.4187)

数据拟合

```python
import matplotlib.pyplot as plt
import numpy as np

x_1 = np.arange(-1.5, 1.6, 0.1)
x = np.arange(-1.5, 1.6, 0.5)
y = np.array([-4.45, 0.45, 0.55, 0.05, -0.44, 0.54, 4.55])
an = np.polyfit(x, y, 3)
print(an)
p1 = np.poly1d(an)
y_line = p1(x_1)
print(p1)
plt.scatter(x, y)
plt.plot(x_1, y_line, color='g')
plt.show()
```

![image-20230818160534444](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230818160534444.png)

OLS数据拟合

```python
import numpy as np
import statsmodels.api as sm  # 实现了类似于二元中的统计模型，比如ols普通最小二乘法

np.random.seed(991)  # 随机数种子
x1 = np.random.normal(0, 0.4, 100)  # 生成符合正态分布的随机数（均值，标准差，所生成随机数的个数）
x2 = np.random.normal(0, 0.6, 100)
x3 = np.random.normal(0, 0.2, 100)
eps = np.random.normal(0, 0.05, 100)
X = np.c_[x1, x2, x3]  # 调用c＿函数来生成自变量的数据的矩阵，按照列进行生成的；100x3的矩阵
beta = [0.1, 0.2, 0.7]  # 生成模拟数据时候的系数的值
y = np.dot(X, beta) + eps  # 点积＋噪声
X_model = sm.add_constant(X)  # addcostant
model = sm.OLS(y, X_model)  # 调用OLs普通最小二乘
results = model.fit()  # fit拟合，主要功能就是进行参数估计，参数估计的主要目的是估计出回归系数，根据参数估计结果来计算统计量，这些统计量主要的目的就是对我们模型的有效性或是显著性水平来
print(results.summary())
```

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.931
Model:                            OLS   Adj. R-squared:                  0.929
Method:                 Least Squares   F-statistic:                     432.6
Date:                Fri, 18 Aug 2023   Prob (F-statistic):           1.29e-55
Time:                        16:13:46   Log-Likelihood:                 152.69
No. Observations:                 100   AIC:                            -297.4
Df Residuals:                      96   BIC:                            -287.0
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0097      0.005     -1.798      0.075      -0.020       0.001
x1             0.0746      0.015      4.941      0.000       0.045       0.105
x2             0.2032      0.009     22.446      0.000       0.185       0.221
x3             0.7461      0.030     25.158      0.000       0.687       0.805
==============================================================================
Omnibus:                        2.535   Durbin-Watson:                   1.771
Prob(Omnibus):                  0.282   Jarque-Bera (JB):                1.864
Skew:                           0.153   Prob(JB):                        0.394
Kurtosis:                       2.405   Cond. No.                         5.54
==============================================================================
```

> coef为系数值

#### 2.1.1概述

插值||拟合：插值，已知有限个数据点，求近似函数；拟合，也是已知有限个数据点，求近似函数，但是只要求总偏差最小。插值函数经过样本点，拟合函数一般基于最小二乘法尽量靠近所有样本点穿过。

#### 2.1.2插值方法

- 拉格朗日插值法

拉格朗日插值多项式：当节点数 n 较大时，拉格朗日插值多项式次数较高，可能收敛不一致，且计算复杂。
高次插值带来误差的震动现象称为 “龙格现象”

- 分段插值法
- 样条插值法

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

plt.rcParams["font.sans-serif"] = ["SimHei"]

plt.rcParams['axes.unicode_minus']=False
x = np.linspace(0, 2.25 * np.pi, 10)
# x = np.linspace(0, 2 * np.pi + np.pi / 4, 10)
y = np.sin(x)
x_new = np.linspace(0, 2 * np.pi + np.pi / 4, 100)
f_linear = interpolate.interp1d(x, y)
tck = interpolate.splrep(x, y)
y_bspline = interpolate.splev(x_new, tck)
# 可视化
plt.xlabel('安培/A')
plt.ylabel('伏特/V')
plt.plot(x, y, "o", label='原始数据')
plt.plot(x_new, f_linear(x_new), label='线性插值')
plt.plot(x_new, y_bspline, label='B-spline插值')
plt.legend()
plt.show()

```

### 2.2 二维插值

### 2.3 OLS拟合

## 三 思路

### 3.1基本方法

- 机理分析
- 测试分析（黑盒分析）

## 四 微分方程
$$
\begin{equation}
\begin{aligned}
\frac{d^{2}x(t)}{dt^{2}}+2\gamma\omega_{0}\frac{dx(t)}{dt}+\omega_{0}^{2}x(t)=0\\
initial\quad{conditions.}
\begin{cases}
&x(0)=1\\
&\left.\frac{dx(t)}{dt}\right|_{t=0}=0
\end{cases}
\end{aligned}
\end{equation}
$$

```python
import sympy  
  
t, omega0, gamma = sympy.symbols("t, omega_0, gamma", positive=True)  
# 变量  
x = sympy.Function('x')  
eq = sympy.diff(x(t), t, 2) + 2*gamma*omega0*sympy.diff(x(t), t) + omega0**2*x(t)  
con = {x(0): 1, sympy.diff(x(t), t).subs(t, 0): 0}  
s = sympy.dsolve(eq, ics=con)  
print(sympy.latex(sympy.simplify(s)))
```
>输出结果

$$
x{\left(t \right)} = \left(- \frac{\gamma}{2 \sqrt{\gamma^{2} - 1}} + \frac{1}{2}\right) e^{- \omega_{0} t \left(\gamma + \sqrt{\gamma^{2} - 1}\right)} + \left(\frac{\gamma}{2 \sqrt{\gamma^{2} - 1}} + \frac{1}{2}\right) e^{\omega_{0} t \left(- \gamma + \sqrt{\gamma^{2} - 1}\right)}
$$

当无解析解时，借助`scipy`中的`intergrate.odeint`求部分性质，辅以可视化结果。
$$
\frac{dy}{dx}=x-y(x)^2
$$
```python
def dynamic_p() -> list:  
    # 物品栏  
    items = [  
        {"name": "水", "weight": 3, "value": 10},  
        {"name": "书", "weight": 1, "value": 3},  
        {"name": "食物", "weight": 2, "value": 9},  
        {"name": "小刀", "weight": 3, "value": 4},  
        {"name": "衣服", "weight": 2, "value": 5},  
        {"name": "手机", "weight": 1, "value": 10}  
    ]    # 约束条件  
    max_capacity = 6  
    dp = [[0] * (max_capacity + 1) for _ in range(len(items) + 1)]  
    for i in range(1, len(items)+1):  
        for j in range(1, max_capacity+1):  
            weight = items[i-1]["weight"]  
            value = items[i-1]["value"]  
            if weight>j:  
                dp[i][j] = dp[i-1][j]  
            else:  
                dp[i][j] = max(value+dp[i-1][j-weight], dp[i-1][j])  
    return dp  
  
  
dp = dynamic_p()  
for i in dp:  
    print(i)  
# 最优解  
print(dp[-1][-1])
```
>输出结果
```python
x=[ 0.   0.5  1.   1.5  2.   2.5  3.   3.5  4.   4.5  5.   5.5  6.   6.5
  7.   7.5  8.   8.5  9.   9.5 10. ]
对应y=[[0.         0.12346145 0.45554459 0.85739248 1.19357597 1.44048259
  1.63015136 1.7893431  1.93113612 2.06132347 2.18278444 2.29723262
  2.40583373 2.50944362 2.60871948 2.70418125 2.79624996 2.88527273
  2.97154005 3.05529819 3.13675822]]
```
![[Pasted image 20230817234513.png]]
```python
import matplotlib.pyplot as plt  
import numpy as np  
from scipy.integrate import odeint  
  
  
# 高阶微分方程需要转化成一阶微分方程  
def fvdp(t, y):  
    dy_1 = y[1]  
    dy_2 = 1000 * (1 - y[0] ** 2) * y[1] - y[0]  
    return [dy_1, dy_2]  
  
  
def solve_ode():  
    x = np.arange(0, 0.25, 0.01)  
    y_0 = [0.0, 2.0]  
    y = odeint(fvdp, y_0, x, tfirst=True)  
    y_1 = plt.plot(x, y[:, 0], label='y')  
    y_2 = plt.plot(x, y[:, 1], label='y’')  
    plt.legend()  
    plt.show()  
  
  
solve_ode()
```
![[Pasted image 20230818000028.png]]
#### 偏微分方程
$$
\begin{cases}\frac{dx}{dt}=2x-3y+3z\\\frac{dy}{dt}=4x-5y+3z\\\frac{dz}{dt}=4x-4y+2z\\x(0)=1,y(0)=2,z(0)=1&\end{cases}
$$
```python
import matplotlib.pyplot as plt  
import numpy as np  
from scipy.integrate import solve_ivp  
  
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  
  
  
#   方程组  
def fun(t, w):  
    x = w[0]  
    y = w[1]  
    z = w[2]  
    return [2 * x - 3 * y + 3 * z, 4 * x - 5 * y + 3 * z, 4 * x - 4 * y + 2 * z]  
  
  
# 初始条件  
y0 = [1, 2, 1]  
yy = solve_ivp(fun, (0, 10), y0, method='RK45', t_eval=np.arange(1, 10, 1))  
t = yy.t  
data = yy.y  
plt.plot(t, data[0, :])  
plt.plot(t, data[1, :])  
plt.plot(t, data[2, :])  
plt.xlabel("时间s")  
plt.show()
```
![[Pasted image 20230818000638.png]]
## 五 差分方程

```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(20, 8))
plt.rcParams['font.sans-serif'] = ['SimHei']
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
b = [0, 0.2, 1.8, 0.8, 0.2]
s = [0.5, 0.8, 0.8, 0.1]
lengthB = len(b)
# 系数矩阵
L = np.zeros((lengthB, lengthB))
x0 = np.full((lengthB, 1), 100)
for i in range(lengthB):
    L[0][i] = b[i]
    if i < lengthB - 1:
        L[i + 1][i] = s[i]
k = np.linspace(0, 30, 31, dtype=int)  # 从(-1,1)均匀取50个点
x = np.empty((lengthB, len(k)))
x_sum = np.empty((lengthB, len(k)))
for j in range(len(k)):
    Xin = np.linalg.matrix_power(L, k[j]) @ x0
    sum = np.sum(Xin)
    for i in range(lengthB):
        x[i][j] = Xin[i][0]
        x_sum[i][j] = Xin[i][0]/sum
for i in range(5):
    ax2.plot(k, x_sum[i], label=f"i={i}")
    ax1.plot(k, x[i], label=f"i={i}")
ax1.set_title(r"第i年龄段的种群数量$x_i(k)$")
ax1.set_xlim((0, 30))
ax1.set_ylim((0, 500))
ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$x_i(k)$')
ax1.grid()
ax2.set_title(r"第i年各年龄分段所占比例$x_{i}^{*}(k)$")
ax2.set_xlim((0, 30))
ax2.set_ylim((0, 0.6))
ax2.set_xlabel(r'$k$')
ax2.set_ylabel(r'$x_{i}^{*}(k)$')
ax2.grid()
plt.show()
```

数据模型图

![image-20230802122454712](C:\Users\lenovo\AppData\Roaming\Typora\typora-user-images\image-20230802122454712.png)

## 六 代数方程

## 七 离散模型

### 7.1 排队论

#### 7.1.1单服务台排队模型算法

```python
import random

n = 100
between = [0 for i in range(100)]
unload = [0 for i in range(100)]
arrive = [0 for i in range(100)]
start = [0 for i in range(100)]
idle = [0 for i in range(100)]
wait = [0 for i in range(100)]
finish = [0 for i in range(100)]
harbor = [0 for i in range(100)]

between[0] = random.randint(15, 145)
unload[0] = random.randint(45, 90)

arrive[0] = between[0]
HARTIME = unload[0]
MAXHAR = unload[0]
WAITIME = 0
MAXWAIT = 0
IDLETIME = arrive[0]

finish[0] = arrive[0] + unload[0]
for i in range(1, 100):
    between[i] = random.randint(15, 145)
    unload[i] = random.randint(45, 90)
    arrive[i] = arrive[i - 1] + between[i]
    timediff = arrive[i] - finish[i - 1]
    if timediff >= 0:
        idle[i] = timediff
        wait[i] = 0
    else:
        wait[i] = -timediff
        idle[i] = 0
    start[i] = arrive[i] + wait[i]
    finish[i] = start[i] + unload[i]
    harbor[i] = wait[i] + unload[i]
    HARTIME += harbor[i]
    if harbor[i] > MAXHAR:
        MAXHAR = harbor[i]
    WAITIME += wait[i]
    IDLETIME += idle[i]
    if wait[i] > MAXWAIT:
        MAXWAIT = wait[i]
HARTIME = HARTIME / n
WAITIME = WAITIME / n
IDLETIME = IDLETIME / finish[99]
print(HARTIME, MAXHAR, WAITIME, MAXWAIT, IDLETIME)
```

### 7.2 多属性决策

#### 7.2.1 多属性决策的要素：

1. 决策目标，备选方案，属性集合：
   1. 决策目标由实际问题决定
   2. 属性集合的选择需要考虑
      1. 选取重要性较强的属性
      2. 属性之间保持独立（低相关性）
      3. 定性和定量属性都需要选择
      4. 属性太多时需要对属性进行分层
2. 属性权重

#### 7.2.2 决策矩阵

|        | 属性1          | 属性2          |
| ------ | -------------- | -------------- |
| 方案1  | 方案1属性1的值 | 方案1属性2的值 |
| 方案 2 | 方案2属性1的值 | 方案2属性2的值 |

> 属性值：定性时需要转换，定量时直接记录

记决策矩阵为$D=(d_{ij})_{m\times n}$
$$
D=
\begin{bmatrix}
d_{11} & d_{12} & \cdots & d_{1n}\\
d_{21} & d_{22} & \cdots & d_{2n}\\
\vdots &\vdots  &        &\vdots \\
d_{m1} & d_{m2} & \cdots & d_{mn}\\
\end{bmatrix}_{m\times n}
$$
例如现有数据如下：
$$
\text{表1 汽车采购中的原始权重}d_{ij}(i=1,2,3,j=1,2,3)
$$

|         | $X_{i}$ | $X_{2}$ | $X_{3}$ |
| ------- | ------- | ------- | ------- |
| $A_{1}$ | 25      | 9       | 7       |
| $A_{2}$ | 18      | 7       | 7       |
| $A_{3}$ | 12      | 5       | 5       |

转化得到决策矩阵$D_{3\times 3}$:
$$
D=
\begin{bmatrix}
25 &  9 &  7\\
18 &  7 &  7\\
12 &  5 &  5\\
\end{bmatrix}
\xRightarrow{统一属性性质}
D=
\begin{bmatrix}
\frac{1}{25} &  9 &  7\\
\frac{1}{18} &  7 &  7\\
\frac{1}{12} &  5 &  5\\
\end{bmatrix}
\left\{
\begin{aligned}
R=
\begin{bmatrix}
0.22360248& 0.42857143& 0.36842105\\
0.31055901& 0.33333333& 0.36842105\\
0.46583851& 0.23809524& 0.26315789\\
\end{bmatrix}
，归一化(1)\\
R=
\begin{bmatrix}
0.48      & 1         & 1         \\
0.66666667& 0.77777778& 1         \\
1         & 0.55555556& 0.71428571\\
\end{bmatrix}
，最大化(2)\\
R=
\begin{bmatrix}
0.37089758& 0.7228974 & 0.63116874\\
0.51513553& 0.56225353& 0.63116874\\
0.77270329& 0.40160966& 0.45083482\\
\end{bmatrix}
，模一化(3)
\end{aligned}
\right.
$$
转化代码：

```python
import numpy as np


def normalization(matrix):
    sum_line = matrix.sum(axis=0)
    matrix_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / sum_line[j]
    return matrix_plus


def maximize(matrix):
    max_line = matrix.max(axis=0)
    matrix_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / max_line[j]
    return matrix_plus


def modularization(matrix):
    absolute_line = np.sqrt(np.sum(np.power(matrix, 2), axis=0))
    matrix_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / absolute_line[j]
    return matrix_plus


np.set_printoptions(suppress=True)
D = np.array([[1 / 25, 9, 7],
              [1 / 18, 7, 7],
              [1 / 12, 5, 5]])
# 归一化
R = normalization(D)
print(R)
# 最大化
R = maximize(D)
print(R)
# 模一化
R = modularization(D)
print(R)
```

- 属性分类：
  - 效益性属性（值正比于效果）
  - 费用型属性（值反比于效果）

标准化流程：

- 统一属性性质：一般将费用型属性转化为效益型属性，可以采用将费用型属性取倒数的方式（常用）或以一个较大数减去属性值

- 将决策矩阵比例尺变换：

  - 归一化：
    $$
    r_{ij}=\frac{d_{ij}}{\sum_{i=1}^{m}d_{ij}}
    $$

  - 最大化：

  $$
  r_{ij}=\frac{d_{ij}}{\max_{i=1,2,3,\cdots m}d_{ij}}
  $$

  - 模一化：

  $$
  r_{ij}=\frac{d_{ij}}{\sqrt{\sum_{i=1}^{m}d_{ij}^{2}}}
  $$

- 

#### 7.2.3 属性权重

各个属性$X_{1},X_{2}\cdots X_{n}$对决策目标的重要程度为权重$\omega_{1},\omega_{2}\cdots\omega_{n}$，满足条件$\sum_{j=1}^{n}{\omega_{j}=1}$，记$\omega=(\omega_{1},\omega_{2}\cdots\omega_{n})^{T}$为权向量。计算属性权重值的方法有主观法，客观法。其中客观法最常用的是：

信息熵法：

将$(1)$式的归一化获得的矩阵$R$各列向量值$(r_{1j},r_{2j}\cdots r_{mj})^{T}(j=1,2\cdots n)$看作信息量的概率分布，将属性$X_{i}$的熵定义为$E_{j}=-k\sum_{i=1}^{m}{r_{ij}\ln{r_{ij}}},\enspace k=\frac{1}{\ln{m}},\enspace j=1,2,\cdots,n$

区分度$F_{i}$为$F_{j}=1-E_{j},0\leq{F_{j}\leq{1}}$,然后定义权重$\omega_{j}=\frac{F_{j}}{\sum_{j=1}^{n}{F_{j}}},\enspace{j=1,2,\cdots{,}n}$。

计算代码

```python
def entropy(matrix):
    k = 1 / np.log(matrix.shape[0])
    ln_matrix = np.log(matrix)
    E = np.empty(matrix.shape[1], dtype=float)
    sum = 0
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            sum += matrix[i][j] * ln_matrix[i][j]
        E[j] = -(k * sum)
        sum = 0
    return E


def distinction(matrix):
    entropy_matrix = entropy(matrix)
    F = np.empty(len(entropy_matrix))
    for i in range(len(entropy_matrix)):
        F[i] = 1 - entropy_matrix[i]
    return F


def weight(matrix):
    distinction_matrix = distinction(matrix)
    omega = np.empty(len(distinction_matrix))
    sum_distinction_matrix = np.sum(distinction_matrix)
    for i in range(len(distinction_matrix)):
        omega[i] = distinction_matrix[i]/sum_distinction_matrix
    return omega
```

#### 7.2.4 综合方法

```python
import numpy as np


def normalization(matrix):
    sum_line = matrix.sum(axis=0)
    matrix_plus = np.empty((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / sum_line[j]
    return matrix_plus


def maximize(matrix):
    max_line = matrix.max(axis=0)
    matrix_plus = np.empty((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / max_line[j]
    return matrix_plus


def modularization(matrix):
    absolute_line = np.sqrt(np.sum(np.power(matrix, 2), axis=0))
    matrix_plus = np.empty((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / absolute_line[j]
    return matrix_plus


def entropy(matrix):
    k = 1 / np.log(matrix.shape[0])
    ln_matrix = np.log(matrix)
    E = np.empty(matrix.shape[1], dtype=float)
    sum = 0
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            sum += matrix[i][j] * ln_matrix[i][j]
        E[j] = -(k * sum)
        sum = 0
    return E


def distinction(matrix):
    entropy_matrix = entropy(matrix)
    F = np.empty(len(entropy_matrix))
    for i in range(len(entropy_matrix)):
        F[i] = 1 - entropy_matrix[i]
    return F


def weight(matrix):
    distinction_matrix = distinction(matrix)
    omega = np.empty(len(distinction_matrix))
    sum_distinction_matrix = np.sum(distinction_matrix)
    for i in range(len(distinction_matrix)):
        omega[i] = distinction_matrix[i] / sum_distinction_matrix
    return omega


def weighted(matrix, weight_matrix, is_sum_weighted):
    V = np.zeros(matrix.shape[0])
    if is_sum_weighted:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                V[i] += matrix[i][j] * weight_matrix[j]
    else:
        for i in range(matrix.shape[0]):
            V[i] = 1
            for j in range(matrix.shape[1]):
                V[i] *= np.power(matrix[i][j], weight_matrix[j])
    return V


def TOPSIS(matrix, weight_matrix):
    V = np.empty((matrix.shape[0], matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            V[i][j] = matrix[i][j] * weight_matrix[j]
    v_min = np.min(V, axis=0)
    v_max = np.max(V, axis=0)
    euclidean_distance_min = np.zeros(len(weight_matrix))
    euclidean_distance_max = np.zeros(len(weight_matrix))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            euclidean_distance_max[i] += np.power((V[i][j] - v_max[j]), 2)
            euclidean_distance_min[i] += np.power((V[i][j] - v_min[j]), 2)
        euclidean_distance_max[i] = np.sqrt(euclidean_distance_max[i])
        euclidean_distance_min[i] = np.sqrt(euclidean_distance_min[i])
    similarity_approach = euclidean_distance_min/(euclidean_distance_max+euclidean_distance_min)
    return normalization(similarity_approach.reshape(len(weight_matrix), 1))


np.set_printoptions(suppress=True)
D = np.array([[1 / 25, 9, 7],
              [1 / 18, 7, 7],
              [1 / 12, 5, 5]])
# 归一化
R = normalization(D)
# print(R)
# print(entropy(R))
# print(distinction(R))
Omega = weight(R)
print("权  重：", Omega)
print("加权和：", weighted(R, Omega, True))
print("加权积：", weighted(R, Omega, False))
R = modularization(D)
print("TOPSIS", TOPSIS(R, Omega).T)
# # 最大化
# R = maximize(D)
# print(R)
# # 模一化
# R = modularization(D)
# print(R)
```

### 7.3 层次分析法

目标层（一个元素）-->准则层-->方案层

正互反阵$A=(a_{ij}),a_{ij}>0,a_{ji}=\frac{1}{a_{ij}}$,或称成对比较阵，举例
$$
A=
\begin{bmatrix}
1 & \frac{1}{2} & \frac{1}{3} & \frac{1}{2}\\
2 & 1 & \frac{1}{2} & 1\\
3 & 2 & 1 & 2\\
2 & 1 & \frac{1}{2} &  1
\end{bmatrix}
$$

一致性检验：$CI=\frac{\lambda-n}{n-1}$当$CI=0$的时候，A是一致阵。反之$CI$值越大，矩阵一致度越低。

随机一致性检验：$RI$

| n    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| $RI$ | 0.58 | 0.90 | 1.12 | 1.24 | 1.32 | 1.41 | 1.45 | 1.49 |

当一致性比率$CR$满足$CR=\frac{CI}{RI}\textless{0.1}$认为A的不一致程度允许范围。

```python
import numpy as np


def normalization(matrix):
    sum_line = matrix.sum(axis=0)
    matrix_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / sum_line[j]
    return matrix_plus


A = np.array([[1, 1 / 2, 1 / 3, 1 / 2],
              [2, 1, 1 / 2, 1],
              [3, 2, 1, 2],
              [2, 1, 1 / 2, 1]])
diag, p = np.linalg.eig(A)
diag = np.real(diag)
p = np.real(p)
max_diag = np.argmax(diag)
CI = (diag[max_diag] - A.shape[0]) / (A.shape[0] - 1)
print(max_diag, diag[max_diag], CI)
print(normalization(p)[max_diag])
```
#### 7.3.1 主成分分析
目的是将数据去中心化。
基本流程：
- 求特征值，特征向量
- 特征值降序排列，从大到小累加，至满足设定到的m值（方差解释）
- 将特征向量聚合成矩阵P
```python
import matplotlib.pyplot as plt  
import sklearn.decomposition as dp  
from sklearn.datasets import load_iris  
  
x, y = load_iris(return_X_y=True)  
pca = dp.PCA(n_components=2)  
reduced_x = pca.fit_transform(x, y)  
red_x, red_y = [], []  
blue_x, blue_y = [], []  
green_x, green_y = [], []  
for i in range(len(reduced_x)):  
    if y[i] == 0:  
        red_x.append(reduced_x[i][0])  
        red_y.append(reduced_x[i][1])  
    elif y[i] == 1:  
        blue_x.append(reduced_x[i][0])  
        blue_y.append(reduced_x[i][1])  
    else:  
        green_x.append(reduced_x[i][0])  
        green_y.append(reduced_x[i][1])  
plt.scatter(red_x, red_y, c='r', marker='x')  
plt.scatter(blue_x, blue_y, c='b', marker='D')  
plt.scatter(green_x, green_y, c='g', marker='.')  
plt.show()
```
![[Pasted image 20230821122904.png]]
#### 7.3.2 因子分析

### 7.4 决策树法

### 7.5 竞赛图

```python
import numpy as np


def normalization(matrix):
    sum_line = matrix.sum(axis=0)
    matrix_plus = np.zeros((matrix.shape[0], matrix.shape[1]))
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            matrix_plus[i][j] = matrix[i][j] / sum_line[j]
    return matrix_plus


time = 25




A = np.array([[0, 1, 0, 1, 0],
              [0, 0, 1, 1, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 1],
              [1, 1, 1, 0, 0]])
s = np.zeros((time, A.shape[0]))

for i in range(time):
    # print(np.linalg.matrix_power(A, i + 1).sum(axis=1))
    s[i] = np.linalg.matrix_power(A, i + 1).sum(axis=1)
print(normalization(s)[time-1])
```

```python
import numpy as np


def rank_price(matrix):
    condition = [0, 300, 350, 400, 450, 500, 600, 700, 800, 900]
    price = [20, 23, 26, 29, 32, 37, 44, 50, 55, 60]

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            sum = 0
            c = np.zeros(len(condition) + 1)
            temp = matrix[i][j]
            for k in range(len(condition)-1, -1, -1):
                temp -= c[k+1]
                c[k] = temp - condition[k]
                if c[k] < 0:
                    c[k] = 0
                    continue
                sum += c[k] * price[k]
            matrix[i][j] = sum
    return matrix


def Dijkstra_all_minpath(matr, start):  # matr为邻接矩阵的数组，start表示起点
    n = len(matr)  # 该图的节点数
    dis = []
    temp = []
    dis.extend(matr[start])  # 添加数组matr的start行元素
    temp.extend(matr[start])  # 添加矩阵matr的start行元素
    temp[start] = np.inf  # 临时数组会把处理过的节点的值变成 \infty
    visited = [start]  # start已处理
    parent = [start] * n  # 用于画路径，记录此路径中该节点的父节点
    while len(visited) < n:
        i = temp.index(min(temp))  # 找最小权值的节点的坐标
        temp[i] = np.inf
        for j in range(n):
            if j not in visited:
                if (dis[i] + matr[i][j]) < dis[j]:
                    dis[j] = temp[j] = dis[i] + matr[i][j]
                    parent[j] = i  # 说明父节点是i
        visited.append(i)  # 该索引已经处理了
        path = []  # 用于画路径
        path.append(str(i + 1))
        k = i
        while (parent[k] != start):  # 找该节点的父节点添加到path，直到父节点是start
            path.append(str(parent[k] + 1))
            k = parent[k]
        path.append(str(start + 1))
        path.reverse()  # path反序产生路径
        # print(str(i + 1) + ':', '->'.join(path))  # 打印路径
    return dis


a = np.array([[     0,    450, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf,      0,   1150,     80, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf,      0, np.inf,   1100, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf,      0,   1200,    202, np.inf, np.inf, np.inf,    720, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,     20, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    195, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    306, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    690,    520, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    170, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    690,     88, np.inf,    160, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    462, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,     70,    320, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,    160, np.inf, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,     70,    290, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0, np.inf, np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0,     30],
              [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,      0]])
b = np.zeros((a.shape[0], a.shape[1]))
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        a[j][i] = a[i][j]
s = np.array([7, 6, 12, 15, 17, 22, 24])
for i in range(a.shape[0]):
    b[i] = Dijkstra_all_minpath(a, i)
b = rank_price(b)
print(b)

```

## 八 概率模型

## 九 统计模型

## 十 博弈模型
## 十一 数学代码编写
### 11.1 分类
- 数值计算
- 图像分析
### 11.2数值计算
- 文件读写
    - json
    - zip 
    - numpy
    - pandas
- 封装函数或面向对象
- 调用现成数据，工具包
### 11.3数据可视化
可视化工具：
- `matplotlib.pyplot as plt`
- `mpl_toolkits`3D绘图
- `seaborn`美化图形0
## 遗传算法
```python
import matplotlib.pyplot as plt  
import numpy as np  
from matplotlib import cm  
  
DNA_SIZE = 24  
POP_SIZE = 200  
CROSSOVER_RATE = 0.8  
MUTATION_RATE = 0.005  
N_GENERATIONS = 50  
X_BOUND = [-3, 3]  
Y_BOUND = [-3, 3]  
  
  
def F(x, y):  
    return 3 * (1 - x) ** 2 * np.exp(-(x ** 2) - (y + 1) ** 2) - 10 * (x / 5 - x ** 3 - y ** 5) * np.exp(  
        -x ** 2 - y ** 2) - 1 / 3 ** np.exp(-(x + 1) ** 2 - y ** 2)  
  
  
def plot_3d(ax):  
    X = np.linspace(*X_BOUND, 100)  
    Y = np.linspace(*Y_BOUND, 100)  
    X, Y = np.meshgrid(X, Y)  
    Z = F(X, Y)  
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)  
    ax.set_zlim(-10, 10)  
    ax.set_xlabel('x')  
    ax.set_ylabel('y')  
    ax.set_zlabel('z')  
    plt.pause(3)  
    plt.show()  
  
  
def get_fitness(pop):  
    x, y = translateDNA(pop)  
    pred = F(x, y)  
    return (pred - np.min(  
        pred)) + 1e-3  # 减去最小的适应度是为了防止适应度出现负数，通过这一步fitness的范围为[0, np.max(pred)-np.min(pred)],最后在加上一个很小的数防止出现为0的适应度  
  
  
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个二进制编码表示的DNA，矩阵的行数为种群数目  
    x_pop = pop[:, 1::2]  # 奇数列表示X  
    y_pop = pop[:, ::2]  # 偶数列表示y  
  
    # pop:(POP_SIZE,DNA_SIZE)*(DNA_SIZE,1) --> (POP_SIZE,1)    x = x_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]  
    y = y_pop.dot(2 ** np.arange(DNA_SIZE)[::-1]) / float(2 ** DNA_SIZE - 1) * (Y_BOUND[1] - Y_BOUND[0]) + Y_BOUND[0]  
    return x, y  
  
  
def crossover_and_mutation(pop, CROSSOVER_RATE=0.8):  
    new_pop = []  
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲  
        child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）  
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉  
            mother = pop[np.random.randint(POP_SIZE)]  # 再种群中选择另一个个体，并将该个体作为母亲  
            cross_points = np.random.randint(low=0, high=DNA_SIZE * 2)  # 随机产生交叉的点  
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因  
        mutation(child)  # 每个后代有一定的机率发生变异  
        new_pop.append(child)  
  
    return new_pop  
  
  
def mutation(child, MUTATION_RATE=0.003):  
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异  
        mutate_point = np.random.randint(0, DNA_SIZE * 2)  # 随机产生一个实数，代表要变异基因的位置  
        child[mutate_point] = child[mutate_point] ^ 1  # 将变异点的二进制为反转  
  
  
def select(pop, fitness):  # nature selection wrt pop's fitness  
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,  
                           p=(fitness) / (fitness.sum()))  
    return pop[idx]  
  
  
def print_info(pop):  
    fitness = get_fitness(pop)  
    max_fitness_index = np.argmax(fitness)  
    print("max_fitness:", fitness[max_fitness_index])  
    x, y = translateDNA(pop)  
    print("最优的基因型：", pop[max_fitness_index])  
    print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))  
  
  
if __name__ == "__main__":  
    fig = plt.figure()  
    ax = plt.axes(projection='3d')  
    plt.ion()  # 将画图模式改为交互模式，程序遇到plt.show不会暂停，而是继续执行  
    plot_3d(ax)  
  
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE * 2))  # matrix (POP_SIZE, DNA_SIZE)  
    for _ in range(N_GENERATIONS):  # 迭代N代  
        x, y = translateDNA(pop)  
        if 'sca' in locals():  
            sca.remove()  
        sca = ax.scatter(x, y, F(x, y), c='black', marker='o')  
        plt.show()  
        plt.pause(0.1)  
        pop = np.array(crossover_and_mutation(pop, CROSSOVER_RATE))  
        # F_values = F(translateDNA(pop)[0], translateDNA(pop)[1])#x, y --> Z matrix  
        fitness = get_fitness(pop)  
        pop = select(pop, fitness)  # 选择生成新的种群  
  
    print_info(pop)  
    plt.ioff()  
    plot_3d(ax)
```
>输出结果

```
max_fitness: 0.025404266456633295
最优的基因型： [1 0 1 1 0 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 0 1 1 1 1 0 0 0 1 1 0 0 0 1 1 0
 1 0 0 0 0 1 1 1 0 1 1]
(x, y): (-0.015161455581274907, 1.569697294813233)
```
>图像

![[Pasted image 20230821164638.png]]
```python
import matplotlib.pyplot as plt  
import numpy as np  
import pandas as pd  
  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签  
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号  
  
  
class B_1_cls:  
    def __init__(self):  
        self.theta_rad = None  
        self.theta_deg = None  
        self.r = None  
        self.allColor = ['b', 'c', 'g', 'k', 'm', 'y', 'purple']  
        self.color = None  
        self.counter = 0  
  
    def _mid_k_func(self, theta_i, theta_j, theta_k_true):  
        theta_rad = None  
        theta_deg = None  
        r = None  
        color = None  
        if theta_k_true - theta_j < 180 and theta_k_true - theta_i < 180:  
            alpha_i = (180 - theta_k_true + theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (180 + theta_k_true - theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) + np.sin(  
                np.radians(-alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            fenmu = np.cos(np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) + np.cos(  
                np.radians(-alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            theta_rad = np.array(pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x if x > 0 else x + np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(180 - theta_j + theta_deg - alpha_j)) / np.sin(np.radians(alpha_j))  
        elif theta_j - theta_k_true > 180 and theta_k_true - theta_i < 180:  
            alpha_i = (180 - theta_k_true + theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (-180 - theta_k_true + theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(-alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) - np.sin(  
                np.radians(-alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            fenmu = np.cos(np.radians(-alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) - np.cos(  
                np.radians(-alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            theta_rad = np.array(pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x if x > 0 else x + np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(-180 - theta_deg + theta_j - alpha_j)) / np.sin(np.radians(alpha_j))  
        elif theta_j - theta_k_true < 180 and theta_k_true - theta_i > 180:  
            alpha_i = (-180 + theta_k_true - theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (180 + theta_k_true - theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j)) - np.sin(  
                np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i))  
            fenmu = np.cos(np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j)) - np.cos(  
                np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i))  
            theta_rad = np.array(  
                pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x + np.pi if x > 0 else x + 2 * np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(180 + theta_deg - theta_j - alpha_j)) / np.sin(np.radians(alpha_j))  
        return theta_rad, theta_deg, r  
  
    def _right_or_left_k_func(self, theta_i, theta_j, theta_k_true):  
        theta_rad = None  
        theta_deg = None  
        r = None  
        if theta_k_true - theta_j < 180 and theta_k_true - theta_i < 180:  
            alpha_i = (180 - theta_k_true + theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (180 - theta_k_true + theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(alpha_j - theta_j)) * np.sin(np.radians(alpha_i)) - np.sin(  
                np.radians(alpha_i - theta_i)) * np.sin(np.radians(alpha_j))  
            fenmu = np.cos(np.radians(alpha_i - theta_i)) * np.sin(np.radians(alpha_j)) - np.cos(  
                np.radians(alpha_j - theta_j)) * np.sin(np.radians(alpha_i))  
            theta_rad = np.array(pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x if x > 0 else x + np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(-theta_j + theta_deg + alpha_j)) / np.sin(np.radians(alpha_j))  
        elif theta_k_true - theta_j < 180 and theta_k_true - theta_i > 180:  
            alpha_i = (-180 + theta_k_true + theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (180 - theta_k_true - theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(-alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) + np.sin(  
                np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            fenmu = np.cos(np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j)) + np.cos(  
                np.radians(alpha_j - theta_j)) * np.sin(np.radians(alpha_i))  
            theta_rad = np.array(  
                pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x + np.pi if x > 0 else x + 2 * np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(theta_deg - theta_j + alpha_j)) / np.sin(np.radians(alpha_j))  
        elif theta_k_true - theta_j > 180 and theta_k_true - theta_j > 180:  
            alpha_i = (-180 + theta_k_true - theta_i) / 2  
            alpha_i = np.arange(alpha_i - 1, alpha_i + 1, 0.1)  
            alpha_j = (-180 + theta_k_true - theta_j) / 2  
            alpha_j = np.arange(alpha_j - 1, alpha_j + 1, 0.1)  
            fenzi = np.sin(np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) - np.sin(  
                np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            fenmu = np.cos(np.radians(alpha_j + theta_j)) * np.sin(np.radians(alpha_i)) - np.cos(  
                np.radians(alpha_i + theta_i)) * np.sin(np.radians(alpha_j))  
            theta_rad = np.array(  
                pd.Series(np.arctan(fenzi / fenmu)).map(lambda x: x + np.pi if x > 0 else x + 2 * np.pi))  
            theta_deg = np.rad2deg(theta_rad)  
            r = 100 * np.sin(np.radians(-180 + theta_deg - theta_j - alpha_j)) / np.sin(np.radians(alpha_j))  
        return theta_rad, theta_deg, r  
  
    def polar_plot(self):  
        Qi = 0  
        Qj = np.arange(40, 360, 40)  
        Qk = np.arange(40, 360, 40)  
        for i, j in enumerate(Qj):  
            ax = plt.subplot(2, 4, i + 1, projection='polar')  
            self.theta_rad = np.array([])  
            self.theta_deg = np.array([])  
            self.r = np.array([])  
            self.color = np.array([])  
            self.counter = 0  
            for k in Qk:  
                if k == j:  
                    continue  
                else:  
                    if j > k:  
                        theta_rad, theta_deg, temp_r = self._mid_k_func(Qi, j, k)  
                        self.theta_rad = np.hstack([self.theta_rad, theta_rad])  
                        self.theta_deg = np.hstack([self.theta_deg, theta_deg])  
                        self.r = np.hstack([self.r, temp_r])  
                        self.color = np.hstack(  
                            [self.color, np.array([self.allColor[self.counter] for _ in range(theta_rad.shape[0])])])  
                    elif j < k:  
                        theta_rad, theta_deg, temp_r = self._right_or_left_k_func(Qi, j, k)  
                        self.theta_rad = np.hstack([self.theta_rad, theta_rad])  
                        self.theta_deg = np.hstack([self.theta_deg, theta_deg])  
                        self.r = np.hstack([self.r, temp_r])  
                        self.color = np.hstack([self.color, np.array([self.allColor[self.counter] for _ in range(theta_rad.shape[0])])])  
                        self.counter += 1  
            ax.scatter(self.theta_rad, self.r, s=10, c=self.color)  
            ax.set_rlim(0, 105)  
            ax.scatter(Qi, 100, s=100, c='r', marker='*')  
            if i == 0:  
                ax.scatter(np.radians(j), 100, s=100, c='r', marker='*', label='极坐标')  
                ax.legend(bbox_to_anchor=(55, 1.35))  
            ax.scatter(np.radians(j), 100, s=100, c='r', marker='*')  
            plt.show()  
  
    def b_2(self, j, k):  
        return self._mid_k_func(0, j, k) if j > k else self._right_or_left_k_func(0, j, k)  
  
  
b_1 = B_1_cls()  
b_1.polar_plot()
```
