# 简单问题定义
对于给定的某条时间序列，找到与其最相似的k条时间序列 (主要是形状相似和异常相似)
# 简单做法v1.0
* 从influxDB中拿到单指标异常检测的score，用score做cross correlation
* 原始值cross correlation (归一化可选)
# 额外需要
* 找两条时间序列之间是否存在一定的(线性)关系