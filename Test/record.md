

## 背景与任务要求

* 背景
大型互联网服务通常会采集大量的时间序列数据（KPI）用以监控其服务质量和可靠性。一些存在调用关系的服务产生的KPI曲线的正常模式和异常模式都可能存在极高的相似关系，从大量给定的KPI时间序列中可以挖掘出可能反映了内部存在调用关系的服务
* 任务要求
给定一定长度的多组KPI曲线，要求找到一个合适指标以衡量任意两曲线间相似程度，对其KPI曲线两两计算相似度后，回复与特定曲线最相似的若干条曲线
## 问题分析

* 给定的KPI数据中对于KPI曲线正常模式原始值和异常得分已经给出，在将异常score曲线作为另一个维度的数据进行曲线相似度判断，本质上和正常模式曲线相似匹配在功能上是相同的，因此本问题抽象为如下问题：
  * 对于给定的若干组曲线，要求找到一个合适指标以衡量任意两曲线间相似程度，对其曲线两两计算相似度后，回复与特定曲线最相似的若干条曲线

## 任务进度
* 第一阶段：
  * 查阅资料并尝试各种已有算法，对算法进行调整
  * 该阶段由于没有统一的benchmark，算法效果验证还是以人眼粗略验证为主
* 第二阶段：
  * 

## 算法效果与部分分析

### 用于测试的算法：

* dtw raw：
  * 基础的fast time warping 算法，取相似度s=$1/(1+distance)$
* dtw shift {x}：
  * 在基础的fast dtw算法基础上，将其中一条曲线A进行竖直方向平移复制x组后分别计算与曲线B的dtw相似度
* dtw map for linear correlation：
  * 进行一次fast dtw计算后，根据dtw拟合路径，将对应的曲线A\B中的坐标对取出（等于重整曲线A\B）映射为新的两条A\B曲线后计算线性回归（linear regression）的回归系数作为相似度
* dtw map for pearson correlation：
  * 进行一次fast dtw计算后，根据dtw拟合路径，将对应的曲线A\B中的坐标对取出（等于重整曲线A\B）映射为新的两条A\B曲线后计算两曲线的Pearson 相关系数作为相似度
* easy pearson correlation：
  * 简单将Pearson correlation作为两曲线相似度
* easy cross correlation：
  * 简单将cross correlation作为两曲线相似度
* easy linear correlation：
  * 将两曲线的值作为x、y值组成散点，将其线性相关系数作为相似度
* max pearson correlation：
  * 将其中一条曲线A进行水平左右方向各平移复制60组(一个允许时间窗长度)后分别计算与曲线B的Pearson correlation，取最大值作为两曲线相似度
* max cross correlation：
  * 将其中一条曲线A进行水平左右方向各平移复制60组(一个允许时间窗长度)后分别计算与曲线B的cross correlation，取最大值作为两曲线相似度
* max linear correlation：
  * 将其中一条曲线A进行水平左右方向各平移复制60组(一个允许时间窗长度)后分别计算与曲线B的linear correlation，取最大值作为两曲线相似度

### 算法效果

* benchmark 定义
  * 前提：假设总共N条曲线中实际可以分为C个类，类别 i 有N(i)条曲线彼此相似，算法针对其中某一个类 k 的某一条曲线s，将N条曲线根据它们对 s 的相似度排序
  * 算法的score rate:
    * 利用算法得到的排序中，对排名为 m 的位置赋予分数 (N-m) ；将k中每条曲线在该排序中的位置的分数求和，并根据k类中曲线数目，对该分数总和进行正规化后的百分比
    * 例如：N=10，类k有3条曲线，排序为 0、4、3，则其得分总和为 (10+6+7)=23，由于含有3条曲线的类的最大得分为(10+9+8)=27，最小得分为(10+1+2)=13，则其score rate为$\frac{23-13}{27-13}\times 100\%=71\%$
    * score rate 反映实际该类别曲线被排序的正确程度
  * 算法的error rate:
    * 假设根据算法得到的排序，将前N(k)个曲线声明为类别k的曲线，那么其中混有不是类别k的概率
    * 例如：类k有5条曲线，计算排序后发现排序前5条曲线中有2条不是类别k的曲线，则其error rate为$40\%$
    * error rate反映根据已知类别曲线数目时，用此算法进行分类时的出错概率
* 对测试集合中100条曲线数据进行各种算法的处理，取算法对100条曲线的score rate和error rate的均值，列表如下：

|algorithm|scoreRate(%)| errorRate(%)|
|:-:| :-: | :-: |
|dtw_s20|99.14|3.18|
|dtw_s10|99.14|3.18|
|dtw_s5|97.40|9.21|
|dtw_s3|94.74|18.15|
|mlc|93.11|21.66|
|dtw_m_l|93.01|24.03|
|dtw_r|91.47|26.87|
|mcc|89.47|28.14|
|mpc|85.26|43.98|
|epc|82.33|48.07|
|ecc|82.33|48.07|
|elc|75.70|48.80|
|dtw_m_p|47.25|64.92|

### score rate 分布（沿纵轴截取50%以上部分显示）

* 横轴：曲线下标
* 纵轴：算法对该曲线的score rate



<table>
<tr><td>dtw_s20</td><td><img src="output/dtw_s20_score_rate.png" height="80"></td>
<td>mcc</td><td><img src="output/mcc_score_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s10</td><td><img src="output/dtw_s10_score_rate.png" height="80"></td>
<td>mpc</td><td><img src="output/mpc_score_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s5</td><td><img src="output/dtw_s5_score_rate.png" height="80"></td>
<td>epc</td><td><img src="output/epc_score_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s3</td><td><img src="output/dtw_s3_score_rate.png" height="80"></td>
<td>ecc</td><td><img src="output/ecc_score_rate.png" height="80"></td>
</tr>
<tr><td>mlc</td><td><img src="output/mlc_score_rate.png" height="80"></td>
<td>elc</td><td><img src="output/elc_score_rate.png" height="80"></td>
</tr>
<tr><td>dtw_m_l</td><td><img src="output/dtw_m_l_score_rate.png" height="80"></td>
<td>dtw_m_p</td><td><img src="output/dtw_m_p_score_rate.png" height="80"></td>
</tr>
<tr><td>dtw_r</td><td><img src="output/dtw_r_score_rate.png" height="80"></td>
</tr>
</table>

### error rate 分布(沿纵轴截取50%以下部分显示)

* 横轴：曲线下标
* 纵轴：算法对该曲线的error rate

<table>
<tr><td>dtw_s20</td><td><img src="output/dtw_s20_error_rate.png" height="80"></td>
<td>mcc</td><td><img src="output/mcc_error_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s10</td><td><img src="output/dtw_s10_error_rate.png" height="80"></td>
<td>mpc</td><td><img src="output/mpc_error_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s5</td><td><img src="output/dtw_s5_error_rate.png" height="80"></td>
<td>epc</td><td><img src="output/epc_error_rate.png" height="80"></td>
</tr>
<tr><td>dtw_s3</td><td><img src="output/dtw_s3_error_rate.png" height="80"></td>
<td>ecc</td><td><img src="output/ecc_error_rate.png" height="80"></td>
</tr>
<tr><td>mlc</td><td><img src="output/mlc_error_rate.png" height="80"></td>
<td>elc</td><td><img src="output/elc_error_rate.png" height="80"></td>
</tr>
<tr><td>dtw_m_l</td><td><img src="output/dtw_m_l_error_rate.png" height="80"></td>
<td>dtw_m_p</td><td><img src="output/dtw_m_p_error_rate.png" height="80"></td>
</tr>
<tr><td>dtw_r</td><td><img src="output/dtw_r_error_rate.png" height="80"></td>
</tr>
</table>
## 分析总结

上面采用的算法主要有3类：

* 传统的基础的序列距离计算方法：如pearson系数、线性系数等（下称easy类方法）
* 在传统计算方法上，对一个序列做窗口滑动扫描取相似度max值的方法（下称max类方法）
* 时间轴上较为宽松能够自适应的dtw类方法

这三类方法从上面可以总结出一些一些信息

### 1. 能够适应水平偏移的算法或者对不适应水平偏移的算法做了水平方向位移尝试之后，匹配效果普遍更好

* 表现：
  * max标记的算法比easy标记算法表现更好
  * dtw(能够适应水平偏移)类算法表现上整体比easy标记的算法表现更好
* 原因分析：
  * 前者算法通过牺牲时间效率，提高对偏移的容忍度，较好理解

### 2. dtw类算法整体而言识别分类效率较高，且与max类shift组数提高能够单调提高分类准确度

* 表现：
  * 见上面图表

* 原因分析
  * （代码演示）
  * 水平方向通过平移复制尝试匹配的算法（max类），处在边缘位置，尤其是平移窗口内的边缘位置的特征会被很容易不可逆地破坏掉
  
    * 当水平复制越多时，危险窗口越大——不再是用时间换准确率，甚至有可能是时间准确率双损失
  * dtw shift类算法只有纵向平移，不会影响曲线的特征信息，这种操作也是可逆的，只是提高了匹配成功的概率
  
    * 当纵向复制越多时，耗费时间当然增加，但是不会对曲线匹配准确率产生较大负面影响（单调性）
  

### 3. 准备探究的部分：dtw 中map方法（dtw_m_l和dtw_m_p）的显著区别的原因

* dtw map类方法，是先利用dtw的路径对两曲线中的对应的点进行匹配，然后利用匹配的点对进行后续一步计算（线性系数或者pearson系数）

* 可以看出，如果直接采用pearson系数和线性相关系数计算曲线相似度，会是pearson系数显著更优
* 但是在dtw map类算法的最后一步，采用了线性系数能够提高相似度判断准确度（相对raw dtw从91%提升到了93%），但是采用pearson系数之后却成了所有算法中表现最差的一组，其中的原因还需要探究