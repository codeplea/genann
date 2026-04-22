/*
 * GENANN - Minimal C Artificial Neural Network
 *
 * Copyright (c) 2015-2018 Lewis Van Winkle
 *
 * http://CodePlea.com
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */


#ifndef GENANN_H
#define GENANN_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GENANN_RANDOM
/* We use the following for uniform random numbers between 0 and 1.
 * If you have a better function, redefine this macro. */
#define GENANN_RANDOM() (((double)rand())/RAND_MAX)
#endif

/* 神经网络结构体前向声明 */
struct genann;

/**
 * @brief 激活函数类型定义
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 激活后的输出值
 */
typedef double (*genann_actfun)(const struct genann *ann, double a);

/**
 * @brief 神经网络核心结构体
 * 
 * 该结构体包含了神经网络的所有参数和数据：
 * - 网络架构参数（输入层、隐藏层、输出层配置）
 * - 激活函数指针
 * - 权重、输出、delta等数据缓冲区
 */
typedef struct genann {
    /* 网络架构参数：输入节点数、隐藏层数、每层隐藏神经元数、输出节点数 */
    int inputs, hidden_layers, hidden, outputs;

    /* 隐藏层神经元使用的激活函数，默认值: genann_act_sigmoid_cached */
    genann_actfun activation_hidden;

    /* 输出层使用的激活函数，默认值: genann_act_sigmoid_cached */
    genann_actfun activation_output;

    /* 所有权重的总数，也是权重缓冲区的大小 */
    int total_weights;

    /* 总神经元数（输入节点 + 所有神经元），也是输出缓冲区的大小 */
    int total_neurons;

    /* 所有权重数组（长度为 total_weights） */
    double *weight;

    /* 存储输入数组和每个神经元的输出（长度为 total_neurons） */
    double *output;

    /* 存储每个隐藏层和输出层神经元的delta值（长度为 total_neurons - inputs）
     * delta用于反向传播时计算梯度
     */
    double *delta;

} genann;

/**
 * @brief 创建并初始化一个新的神经网络
 * @param inputs 输入节点数量
 * @param hidden_layers 隐藏层数量（0表示没有隐藏层，即单层感知机）
 * @param hidden 每个隐藏层的神经元数量
 * @param outputs 输出节点数量
 * @return 初始化后的神经网络指针，失败返回NULL
 */
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs);

/**
 * @brief 从文件读取并创建神经网络
 * @param in 已打开的文件指针
 * @return 从文件加载的神经网络指针，失败返回NULL
 */
genann *genann_read(FILE *in);

/**
 * @brief 随机初始化神经网络的权重
 * @param ann 神经网络指针
 * 
 * 权重范围: -0.5 到 0.5
 * genann_init()会自动调用此函数
 */
void genann_randomize(genann *ann);

/**
 * @brief 创建神经网络的深拷贝
 * @param ann 源神经网络指针
 * @return 新的神经网络副本，失败返回NULL
 */
genann *genann_copy(genann const *ann);

/**
 * @brief 释放神经网络占用的内存
 * @param ann 要释放的神经网络指针
 */
void genann_free(genann *ann);

/**
 * @brief 执行前向传播算法计算神经网络的输出
 * @param ann 神经网络指针
 * @param inputs 输入数据数组
 * @return 输出数组指针（长度为 ann->outputs）
 * 
 * 前向传播：将输入信号从输入层经过隐藏层传递到输出层的过程
 */
double const *genann_run(genann const *ann, double const *inputs);

/**
 * @brief 执行一次反向传播训练更新
 * @param ann 神经网络指针
 * @param inputs 输入数据
 * @param desired_outputs 期望输出（目标值）
 * @param learning_rate 学习率（控制每次更新的步长）
 * 
 * 反向传播：根据误差计算梯度，逐层更新权重的训练算法
 */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate);

/**
 * @brief 将神经网络保存到文件
 * @param ann 神经网络指针
 * @param out 已打开的输出文件指针
 * 
 * 保存格式：先保存网络架构参数，再保存所有权重值
 */
void genann_write(genann const *ann, FILE *out);

/**
 * @brief 初始化Sigmoid激活函数的查找表
 * @param ann 神经网络指针
 * 
 * 用于加速Sigmoid函数计算，通过预计算值进行查表
 */
void genann_init_sigmoid_lookup(const genann *ann);

/**
 * @brief 标准Sigmoid激活函数
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 1.0 / (1 + exp(-a))
 * 
 * Sigmoid函数将任意输入压缩到(0, 1)区间
 * 公式: σ(x) = 1 / (1 + e^(-x))
 */
double genann_act_sigmoid(const genann *ann, double a);

/**
 * @brief 使用查找表的快速Sigmoid激活函数
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 近似的Sigmoid值
 * 
 * 通过预计算的查找表加速计算，精度略低但速度更快
 */
double genann_act_sigmoid_cached(const genann *ann, double a);

/**
 * @brief 阈值激活函数
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 1 如果a>0，否则0
 * 
 * 也称为阶跃函数，是最早的激活函数之一
 */
double genann_act_threshold(const genann *ann, double a);

/**
 * @brief 线性激活函数
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 输入值本身
 * 
 * 线性激活函数通常用于回归问题的输出层
 */
double genann_act_linear(const genann *ann, double a);


#ifdef __cplusplus
}
#endif

#endif /*GENANN_H*/
