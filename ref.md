# 神经网络学习参考资源

本文档收集了一些高质量的神经网络入门学习资源，帮助你理解 Genann 项目的原理和实现。

---

## 📚 基础概念

### 神经网络基础

| 资源 | 描述 |
|------|------|
| [神经网络简介 - 维基百科](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C) | 神经网络的基本概念、历史和应用 |
| [神经元 - 维基百科](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E5%85%83) | 人工神经元的工作原理 |
| [感知机 - 维基百科](https://zh.wikipedia.org/wiki/%E6%84%9F%E7%9F%A5%E6%9C%BA) | 最简单的神经网络模型 |

### 核心算法

| 资源 | 描述 |
|------|------|
| [反向传播算法 - 维基百科](https://zh.wikipedia.org/wiki/%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD) | Genann 核心训练算法详解 |
| [梯度下降法 - 维基百科](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D) | 神经网络训练的基础优化算法 |
| [随机梯度下降 - 维基百科](https://zh.wikipedia.org/wiki/%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E4%B8%8B%E9%99%8D) | 实际训练中最常用的变体 |

### 激活函数

| 资源 | 描述 |
|------|------|
| [激活函数 - 维基百科](https://zh.wikipedia.org/wiki/%E6%BF%80%E6%B4%BB%E5%87%BD%E6%95%B0) | 激活函数的作用和常见类型 |
| [Sigmoid 函数 - 维基百科](https://zh.wikipedia.org/wiki/S%E5%BD%A2%E5%87%BD%E6%95%B0) | Genann 默认使用的激活函数 |

---

## 🎯 理解 Genann 代码

### 前向传播 (Forward Propagation)

**学习要点：
1. **加权求和：每个神经元接收上一层所有神经元的输出乘以权重，再加上偏置
2. **激活函数：将加权和通过激活函数转换为输出

**参考资源：**
- [前向传播算法详解](https://zhuanlan.zhihu.com/p/370536304) - 知乎专栏
- [神经网络前向传播过程](https://blog.csdn.net/qq_32241189/article/details/80460946) - CSDN 博客

### 反向传播 (Backpropagation)

**学习要点：**
1. **计算误差：输出层误差 = 期望输出 - 实际输出
2. **计算 delta：**
   - 输出层 delta = 误差 × 激活函数导数
   - 隐藏层 delta = 下一层 delta 加权和 × 激活函数导数
3. **更新权重：** Δ权重 = 学习率 × delta × 输入

**参考资源：**
- [反向传播算法详解](https://zhuanlan.zhihu.com/p/38190718) - 知乎专栏
- [一文弄懂神经网络中的反向传播法](https://www.cnblogs.com/charlotte77/p/5629899.html) - 博客园

**Genann 中的反向传播实现：

在 `genann.c` 的 `genann_train()` 函数中：

```c
// 第1步：前向传播（计算各层输出）
genann_run(ann, inputs);

// 第2步：计算输出层 delta
// delta = (target - output) * output * (1 - output)
// 这里的 output * (1 - output) 是 sigmoid 函数的导数！

// 第3步：反向计算隐藏层 delta
// delta = Σ(next_delta * weight) * output * (1 - output)

// 第4步：更新权重
// Δweight = learning_rate * delta * input
```

---

## 📖 推荐教程

### 中文教程

| 资源 | 描述 |
|------|------|
| [神经网络入门 - 李宏毅](https://www.bilibili.com/video/BV1JE411g7XF) | 台湾大学李宏毅教授的深度学习课程（B站视频） |
| [深度学习入门：基于Python的理论与实现](https://book.douban.com/subject/30270953/) | 斋藤康毅著，深入浅出的入门书 |
| [神经网络与深度学习](https://nndl.github.io/) | 免费在线中文书，理论与实践结合 |

### 英文教程（深度好文）

| 资源 | 描述 |
|------|------|
| [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) | Michael Nielsen 经典在线书，详细解释反向传播 |
| [3Blue1Brown 神经网络系列](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_6700YeSf8EmW) | 可视化讲解神经网络（B站有搬运翻译） |
| [CS231n 课程笔记](https://cs231n.github.io/neural-networks-1/) | 斯坦福大学计算机视觉课程 |

---

## 🔬 Genann 代码结构解析

### 文件结构

```
genann/
├── genann.h      # 头文件，定义数据结构和函数声明
├── genann.c      # 实现文件，核心算法实现
├── example1.c    # 示例：反向传播学习 XOR
├── example2.c    # 示例：随机搜索学习 XOR
├── example3.c    # 示例：从文件加载网络
├── example4.c    # 示例：IRIS 数据集分类
└── test.c        # 测试代码
```

### 核心数据结构

```c
typedef struct genann {
    int inputs;           // 输入节点数
    int hidden_layers;    // 隐藏层数
    int hidden;           // 每个隐藏层的神经元数
    int outputs;          // 输出节点数
    
    double *weight;       // 所有权重（连续内存块）
    double *output;       // 各层输出
    double *delta;        // 反向传播用的 delta 值
    
    // 激活函数指针
    genann_actfun activation_hidden;  // 隐藏层激活函数
    genann_actfun activation_output;  // 输出层激活函数
} genann;
```

### 关键函数

| 函数 | 作用 |
|------|------|
| `genann_init()` | 创建并初始化神经网络 |
| `genann_run()` | 执行前向传播，计算输出 |
| `genann_train()` | 执行一次反向传播训练 |
| `genann_read/write()` | 从文件读取/保存网络 |
| `genann_randomize()` | 随机初始化权重 |
| `genann_free()` | 释放内存 |

---

## 💡 学习建议

### 第1步：理解概念
1. 理解什么是神经元、权重、偏置
2. 理解前向传播的计算过程
3. 理解激活函数的作用

### 第2步：理解训练
1. 理解误差如何计算
2. 理解梯度下降的思想
3. 理解反向传播如何计算梯度

### 第3步：阅读代码
1. 先看 `genann.h` 理解数据结构
2. 再看 `genann.c` 中的 `genann_run()` 理解前向传播
3. 最后看 `genann_train()` 理解反向传播

### 第4步：动手实践
1. 编译运行 `example1.c` 学习 XOR
2. 修改网络结构，观察效果
3. 尝试用 Genann 解决其他问题

---

## 🧮 数学基础

### 你需要了解的数学概念

| 概念 | 说明 |
|------|------|
| 导数/偏导数 | 理解梯度下降的基础 |
| 链式法则 | 反向传播算法的数学基础 |
| 矩阵/向量运算 | 神经网络的并行计算视角 |
| 概率统计 | 理解损失函数和评估指标 |

### 链式法则示例

对于复合函数求导：
```
如果 y = f(g(x))
那么 dy/dx = f'(g(x)) * g'(x)
```

在反向传播中：
- 误差对权重的偏导 = 误差对输出的偏导 × 输出对加权和的偏导 × 加权和对权重的偏导
- 即：∂E/∂w = δ * input

---

## 🔗 更多资源

### 相关项目
- [FANN](http://leenissen.dk/fann/wp/) - 功能更完整的 C 语言神经网络库
- [Tinn](https://github.com/glouw/tinn) - 更小型的单隐藏层神经网络库

### 数据集
- [IRIS 数据集](https://archive.ics.uci.edu/ml/datasets/Iris) - 经典分类数据集（Genann 示例使用）
- [MNIST](http://yann.lecun.com/exdb/mnist/) - 手写数字识别数据集

### 工具
- [TensorFlow Playground](https://playground.tensorflow.org/) - 可视化神经网络训练过程
- [Neural Network Zoo](https://www.asimovinstitute.org/neural-network-zoo/) - 各种神经网络结构图

---

*最后更新: 2026-04-22*
