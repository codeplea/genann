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

#include "genann.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * 激活函数宏定义
 * 如果未定义genann_act，则使用间接调用方式（通过函数指针调用激活函数）
 * 间接调用允许在运行时动态切换激活函数
 */
#ifndef genann_act
#define genann_act_hidden genann_act_hidden_indirect
#define genann_act_output genann_act_output_indirect
#else
#define genann_act_hidden genann_act
#define genann_act_output genann_act
#endif

/* Sigmoid查找表的大小，用于缓存计算以提高性能 */
#define LOOKUP_SIZE 4096

/**
 * @brief 隐藏层激活函数的间接调用包装器
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 激活后的输出值
 * 
 * 通过函数指针调用实际的激活函数，实现运行时可配置
 */
double genann_act_hidden_indirect(const struct genann *ann, double a) {
    return ann->activation_hidden(ann, a);
}

/**
 * @brief 输出层激活函数的间接调用包装器
 * @param ann 神经网络指针
 * @param a 输入值
 * @return 激活后的输出值
 */
double genann_act_output_indirect(const struct genann *ann, double a) {
    return ann->activation_output(ann, a);
}

/* Sigmoid查找表的定义域范围 [-15.0, 15.0]
 * 超出此范围的Sigmoid值已经趋近于0或1，可以直接使用边界值
 */
const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;           /* 查找表中每个元素对应的输入间隔 */
double lookup[LOOKUP_SIZE];/* Sigmoid函数值的查找表 */

/* GCC内置函数，用于分支预测优化 */
#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)   /* 提示编译器该分支很可能执行 */
#define unlikely(x)     __builtin_expect(!!(x), 0)   /* 提示编译器该分支不太可能执行 */
#define unused          __attribute__((unused))        /* 标记参数未使用，避免编译器警告 */
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* 禁用MSVC对fscanf的安全警告 */
#endif


/**
 * @brief 标准Sigmoid激活函数实现
 * @param ann 神经网络指针（未使用）
 * @param a 输入值
 * @return Sigmoid函数值
 * 
 * Sigmoid函数公式: σ(x) = 1 / (1 + e^(-x))
 * 输出范围: (0, 1)
 * 
 * 对于极小或极大的输入值，直接返回0或1（避免计算exp时溢出）
 */
double genann_act_sigmoid(const genann *ann unused, double a) {
    if (a < -45.0) return 0;  /* e^45 已经远大于 double 能表示的精度 */
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

/**
 * @brief 初始化Sigmoid查找表
 * @param ann 神经网络指针
 * 
 * 预计算Sigmoid在[-15.0, 15.0]区间的值，
 * 存储在lookup数组中，供genann_act_sigmoid_cached快速查找
 * 
 * 这是一种空间换时间的优化策略
 */
void genann_init_sigmoid_lookup(const genann *ann) {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = genann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

/**
 * @brief 使用查找表的快速Sigmoid激活函数
 * @param ann 神经网络指针（未使用）
 * @param a 输入值
 * @return 近似的Sigmoid函数值
 * 
 * 通过查找预计算的表来加速Sigmoid计算
 * 适用于对精度要求不高但追求速度的场景
 * 
 * 速度比直接计算exp()快得多，但有微小的精度损失
 */
double genann_act_sigmoid_cached(const genann *ann unused, double a) {
    assert(!isnan(a));  /* 断言输入不是NaN */

    /* 边界检查：超出查找表范围的使用边界值 */
    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    /* 计算查找表索引，+0.5是为了四舍五入 */
    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* 浮点数精度问题的安全检查 */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

/**
 * @brief 线性激活函数
 * @param ann 神经网络指针（未使用）
 * @param a 输入值
 * @return 输入值本身
 * 
 * 线性激活函数: f(x) = x
 * 常用于：
 * 1. 回归问题的输出层（需要预测连续值）
 * 2. 某些特殊网络结构
 */
double genann_act_linear(const struct genann *ann unused, double a) {
    return a;
}

/**
 * @brief 阈值激活函数（阶跃函数）
 * @param ann 神经网络指针（未使用）
 * @param a 输入值
 * @return 1 如果a>0，否则0
 * 
 * 阈值函数公式: f(x) = 1 如果 x > 0，否则 0
 * 
 * 这是最早的神经网络激活函数（感知机使用）
 * 缺点：不可导，无法使用梯度下降法训练
 */
double genann_act_threshold(const struct genann *ann unused, double a) {
    return a > 0;
}

/**
 * @brief 初始化并创建一个新的神经网络
 * @param inputs 输入节点数量
 * @param hidden_layers 隐藏层数量（可以是0，表示没有隐藏层）
 * @param hidden 每个隐藏层的神经元数量
 * @param outputs 输出节点数量
 * @return 初始化后的神经网络指针，内存分配失败返回NULL
 * 
 * 神经网络结构说明：
 * - 输入层：接收外部输入数据
 * - 隐藏层：进行特征提取和变换（可选，可以多层）
 * - 输出层：产生最终输出
 * 
 * 权重计算说明：
 * - 每个神经元都有一个偏置(bias)，相当于额外的输入-1.0
 * - 所以每层的权重数 = (上一层神经元数 + 1) * 当前层神经元数
 */
genann *genann_init(int inputs, int hidden_layers, int hidden, int outputs) {
    /* 参数有效性检查 */
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;

    /**
     * 计算隐藏层权重总数
     * 公式解析：
     * - 第一个隐藏层：(inputs + 1) * hidden （+1是因为偏置）
     * - 后续隐藏层：(hidden + 1) * hidden 每层
     * - 总共有 (hidden_layers-1) 个后续隐藏层
     */
    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    
    /**
     * 计算输出层权重总数
     * - 如果有隐藏层：输入来自最后一个隐藏层 (hidden + 1)
     * - 如果没有隐藏层：输入直接来自输入层 (inputs + 1)
     */
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    /* 计算总神经元数：输入节点 + 所有隐藏层神经元 + 输出层神经元 */
    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /**
     * 内存分配说明（一次性分配连续内存块）：
     * 1. genann 结构体本身
     * 2. weight数组: total_weights 个 double
     * 3. output数组: total_neurons 个 double
     * 4. delta数组: (total_neurons - inputs) 个 double（输入层没有delta）
     * 
     * 这种连续内存分配方式便于：
     * - 内存管理简单（一次malloc，一次free）
     * - 缓存友好（数据局部性好）
     * - 便于其他优化算法（如遗传算法）直接操作权重数组
     */
    const int size = sizeof(genann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    /* 初始化网络架构参数 */
    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->total_weights = total_weights;
    ret->total_neurons = total_neurons;

    /**
     * 设置各数据缓冲区的指针
     * 内存布局：
     * [genann结构体][weight数组][output数组][delta数组]
     */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    /* 随机初始化权重 */
    genann_randomize(ret);

    /* 设置默认激活函数为带缓存的Sigmoid */
    ret->activation_hidden = genann_act_sigmoid_cached;
    ret->activation_output = genann_act_sigmoid_cached;

    /* 初始化Sigmoid查找表 */
    genann_init_sigmoid_lookup(ret);

    return ret;
}


/**
 * @brief 从文件读取并创建神经网络
 * @param in 已打开的文件指针
 * @return 加载的神经网络指针，失败返回NULL
 * 
 * 文件格式说明：
 * 第一行：inputs hidden_layers hidden outputs（四个整数）
 * 后续：total_weights 个浮点数（权重值，科学计数法格式）
 * 
 * 使用示例：
 * @code
 * FILE *f = fopen("network.ann", "r");
 * genann *ann = genann_read(f);
 * fclose(f);
 * @endcode
 */
genann *genann_read(FILE *in) {
    int inputs, hidden_layers, hidden, outputs;
    int rc;

    /* 读取网络架构参数 */
    errno = 0;
    rc = fscanf(in, "%d %d %d %d", &inputs, &hidden_layers, &hidden, &outputs);
    if (rc < 4 || errno != 0) {
        perror("fscanf");
        return NULL;
    }

    /* 创建神经网络结构 */
    genann *ann = genann_init(inputs, hidden_layers, hidden, outputs);

    /* 读取所有权重值 */
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        errno = 0;
        rc = fscanf(in, " %le", ann->weight + i);  /* %le 读取double类型 */
        if (rc < 1 || errno != 0) {
            perror("fscanf");
            genann_free(ann);  /* 读取失败，释放已分配的内存 */
            return NULL;
        }
    }

    return ann;
}


/**
 * @brief 创建神经网络的深拷贝
 * @param ann 源神经网络指针
 * @return 新的神经网络副本，失败返回NULL
 * 
 * 深拷贝会复制：
 * - 网络架构参数
 * - 所有权重值
 * - 所有输出缓存
 * - 所有delta缓存
 * 
 * 注意：返回的新网络需要单独调用 genann_free() 释放
 */
genann *genann_copy(genann const *ann) {
    /* 计算需要复制的内存大小（与genann_init中的计算相同） */
    const int size = sizeof(genann) + sizeof(double) * (ann->total_weights + ann->total_neurons + (ann->total_neurons - ann->inputs));
    genann *ret = malloc(size);
    if (!ret) return 0;

    /* 一次性复制所有数据（连续内存块） */
    memcpy(ret, ann, size);

    /**
     * 重要：需要重新设置指针！
     * 因为源网络中的指针指向源网络的内存地址，
     * 新网络需要指向自己的内存地址
     */
    ret->weight = (double*)((char*)ret + sizeof(genann));
    ret->output = ret->weight + ret->total_weights;
    ret->delta = ret->output + ret->total_neurons;

    return ret;
}


/**
 * @brief 随机初始化神经网络的权重
 * @param ann 神经网络指针
 * 
 * 权重初始化范围：[-0.5, 0.5]
 * 
 * 为什么选择这个范围？
 * - 初始权重不宜过大，否则Sigmoid等激活函数会饱和（梯度消失）
 * - 初始权重不宜过小，否则信号传递过程中会逐渐消失
 * - [-0.5, 0.5]是一个经验值，适用于大多数情况
 * 
 * 注意：genann_init() 会自动调用此函数
 */
void genann_randomize(genann *ann) {
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        double r = GENANN_RANDOM();  /* 生成 [0.0, 1.0] 范围的随机数 */
        /* 转换为 [-0.5, 0.5] 范围 */
        ann->weight[i] = r - 0.5;
    }
}


/**
 * @brief 释放神经网络占用的内存
 * @param ann 要释放的神经网络指针
 * 
 * 由于genann使用连续内存分配方式，
 * 只需要一次free()调用即可释放所有内存
 * 
 * 使用示例：
 * @code
 * genann *ann = genann_init(2, 1, 3, 1);
 * // ... 使用网络 ...
 * genann_free(ann);  // 使用完毕后释放
 * @endcode
 */
void genann_free(genann *ann) {
    /* weight, output, delta 指针都指向同一个内存块，只需释放一次 */
    free(ann);
}


/**
 * @brief 执行前向传播，计算神经网络的输出
 * @param ann 神经网络指针
 * @param inputs 输入数据数组（长度为 ann->inputs）
 * @return 输出数组指针（长度为 ann->outputs）
 * 
 * 【前向传播算法原理】
 * 前向传播是将输入信号从输入层经过隐藏层传递到输出层的过程。
 * 每个神经元的计算：
 *   1. 加权求和：sum = Σ(weight[i] * input[i]) + bias
 *   2. 激活函数：output = activation(sum)
 * 
 * 【偏置(bias)的实现方式】
 * 在Genann中，偏置被实现为一个额外的权重，对应固定输入-1.0：
 *   - 每个神经元的第一个权重 *w++ * -1.0 就是偏置项
 *   - 这样可以将偏置和权重统一处理，简化代码
 * 
 * 【权重内存布局】
 * 每一层的权重按以下顺序存储：
 *   [偏置权重][输入1的权重][输入2的权重]...[输入N的权重]
 *   对每个神经元重复上述布局
 * 
 * @note 返回的指针指向内部缓冲区，不要free，数据会被下次调用覆盖
 */
double const *genann_run(genann const *ann, double const *inputs) {
    double const *w = ann->weight;           /* 当前处理的权重位置 */
    double *o = ann->output + ann->inputs;   /* 当前输出位置（跳过输入层） */
    double const *i = ann->output;            /* 当前层的输入（上一层的输出） */

    /**
     * 将输入数据复制到output数组的开始位置
     * 这样输入层就可以像其他层一样处理，不需要特殊代码
     * output数组布局：[输入值][隐藏层输出][输出层输出]
     */
    memcpy(ann->output, inputs, sizeof(double) * ann->inputs);

    int h, j, k;  /* h:隐藏层索引, j:当前层神经元索引, k:上一层神经元索引 */

    /**
     * 情况1：没有隐藏层（单层感知机）
     * 输入层直接连接到输出层
     */
    if (!ann->hidden_layers) {
        double *ret = o;  /* 记住输出层起始位置，用于返回 */
        for (j = 0; j < ann->outputs; ++j) {
            /* 第一个权重是偏置：bias = weight * (-1.0) */
            double sum = *w++ * -1.0;
            /* 累加所有输入的加权和 */
            for (k = 0; k < ann->inputs; ++k) {
                sum += *w++ * i[k];
            }
            /* 应用激活函数，存储输出 */
            *o++ = genann_act_output(ann, sum);
        }
        return ret;
    }

    /**
     * 情况2：有隐藏层
     * 第一步：计算第一个隐藏层
     */
    for (j = 0; j < ann->hidden; ++j) {
        double sum = *w++ * -1.0;  /* 偏置项 */
        for (k = 0; k < ann->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = genann_act_hidden(ann, sum);
    }

    /* 移动输入指针到第一个隐藏层的输出 */
    i += ann->inputs;

    /**
     * 第二步：计算后续隐藏层（如果有多个隐藏层）
     * h从1开始，因为第一个隐藏层已经计算过了
     */
    for (h = 1; h < ann->hidden_layers; ++h) {
        for (j = 0; j < ann->hidden; ++j) {
            double sum = *w++ * -1.0;  /* 偏置项 */
            for (k = 0; k < ann->hidden; ++k) {
                sum += *w++ * i[k];  /* 输入来自上一个隐藏层 */
            }
            *o++ = genann_act_hidden(ann, sum);
        }
        i += ann->hidden;  /* 移动到下一个隐藏层的输出 */
    }

    /* 记住输出层的起始位置 */
    double const *ret = o;

    /**
     * 第三步：计算输出层
     */
    for (j = 0; j < ann->outputs; ++j) {
        double sum = *w++ * -1.0;  /* 偏置项 */
        for (k = 0; k < ann->hidden; ++k) {
            sum += *w++ * i[k];  /* 输入来自最后一个隐藏层 */
        }
        *o++ = genann_act_output(ann, sum);
    }

    /* 调试用断言：确保使用了所有的权重，计算了所有的输出 */
    assert(w - ann->weight == ann->total_weights);
    assert(o - ann->output == ann->total_neurons);

    return ret;
}


/**
 * @brief 执行一次反向传播训练
 * @param ann 神经网络指针
 * @param inputs 输入数据
 * @param desired_outputs 期望输出（目标值）
 * @param learning_rate 学习率（控制每次更新的步长）
 * 
 * 【反向传播算法原理】
 * 反向传播是一种用于训练神经网络的监督学习算法。
 * 核心思想：
 *   1. 首先进行前向传播，计算网络输出
 *   2. 计算输出与目标值之间的误差
 *   3. 从输出层开始，向后（反向）计算每个神经元对误差的"责任"（delta）
 *   4. 根据delta更新权重，减小误差
 * 
 * 【数学基础】
 * 使用梯度下降法最小化误差。对于平方误差：
 *   Error = 0.5 * Σ(target - output)²
 * 
 * 权重更新规则（梯度下降）：
 *   Δweight = -learning_rate * ∂Error/∂weight
 * 
 * 【Sigmoid激活函数的导数】
 * Sigmoid函数: σ(x) = 1/(1+e^(-x))
 * 其导数: σ'(x) = σ(x) * (1 - σ(x))
 * 这就是代码中 *o * (1.0 - *o) 的来源！
 * 
 * 【Delta的含义】
 * delta 表示神经元对最终误差的"贡献"程度：
 * - 输出层: delta = (target - output) * derivative
 * - 隐藏层: delta = (Σ(next_layer_delta * weight)) * derivative
 */
void genann_train(genann const *ann, double const *inputs, double const *desired_outputs, double learning_rate) {
    /**
     * 第一步：必须先执行前向传播
     * 反向传播需要用到前向传播计算出的各层输出值
     */
    genann_run(ann, inputs);

    int h, j, k;  /* h:层索引, j:当前层神经元, k:相关神经元 */

    /**
     * 第二步：计算输出层的delta
     * 
     * 输出层delta公式（使用Sigmoid激活函数）：
     *   δ = (target - output) * output * (1 - output)
     * 
     * 为什么这样计算？
     * - (target - output) 是误差
     * - output * (1 - output) 是Sigmoid函数的导数
     * - 两者相乘得到该神经元对误差的"责任"
     * 
     * 如果使用线性激活函数，导数为1，所以：
     *   δ = (target - output)
     */
    {
        /* 定位到输出层的起始位置 */
        double const *o = ann->output + ann->inputs + ann->hidden * ann->hidden_layers; /* 第一个输出值 */
        double *d = ann->delta + ann->hidden * ann->hidden_layers;                        /* 第一个delta */
        double const *t = desired_outputs;                                                  /* 第一个目标值 */

        /* 根据激活函数类型选择不同的delta计算方式 */
        if (genann_act_output == genann_act_linear ||
                ann->activation_output == genann_act_linear) {
            /* 线性激活函数：delta = 误差 = target - output */
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            /**
             * Sigmoid激活函数（或其他非线性激活函数）：
             * delta = 误差 * 激活函数导数
             *       = (target - output) * output * (1 - output)
             * 
             * 注意：这里利用了Sigmoid导数的特性 σ'(x) = σ(x)*(1-σ(x))
             */
            for (j = 0; j < ann->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }

    /**
     * 第三步：计算隐藏层的delta（从最后一层隐藏层开始，反向传播）
     * 
     * 隐藏层delta公式：
     *   δ_hidden = (Σ(δ_next * weight)) * output * (1 - output)
     * 
     * 含义：
     * - 隐藏层神经元的delta由下一层所有神经元的delta加权求和得到
     * - 权重就是连接这两层的权重值
     * - 同样需要乘以激活函数的导数
     * 
     * 这就是"反向传播"名称的由来：误差从输出层向后传播
     */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* 定位当前隐藏层的输出和delta */
        double const *o = ann->output + ann->inputs + (h * ann->hidden);
        double *d = ann->delta + (h * ann->hidden);

        /* 定位下一层的delta（可能是隐藏层或输出层） */
        double const * const dd = ann->delta + ((h+1) * ann->hidden);

        /* 定位连接到下一层的权重 */
        double const * const ww = ann->weight + ((ann->inputs+1) * ann->hidden) + ((ann->hidden+1) * ann->hidden * (h));

        for (j = 0; j < ann->hidden; ++j) {

            double delta = 0;

            /**
             * 累加下一层所有神经元的delta乘以连接权重
             * 这就是"误差反向传播"的核心计算
             */
            for (k = 0; k < (h == ann->hidden_layers-1 ? ann->outputs : ann->hidden); ++k) {
                const double forward_delta = dd[k];
                const int windex = k * (ann->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            /**
             * 乘以激活函数的导数
             * delta = 加权和 * output * (1 - output)
             */
            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
        }
    }

    /**
     * 第四步：更新输出层的权重
     * 
     * 权重更新公式：
     *   Δweight = learning_rate * delta * input
     *   
     * 注意：这里实际使用的是 +learning_rate，为什么？
     * 因为：
     *   - 标准梯度下降：Δw = -η * ∂E/∂w
     *   - ∂E/∂w = -delta * input（因为误差函数定义的原因）
     *   - 所以：Δw = η * delta * input
     * 
     * 对于偏置权重（第一个权重）：
     *   偏置的"输入"固定为-1.0
     *   所以：Δbias = learning_rate * delta * (-1.0)
     */
    {
        /* 定位输出层的delta */
        double const *d = ann->delta + ann->hidden * ann->hidden_layers;

        /* 定位输出层的权重起始位置 */
        double *w = ann->weight + (ann->hidden_layers
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * ann->hidden * (ann->hidden_layers-1))
                : (0));

        /* 定位上一层的输出（作为当前层的输入） */
        double const * const i = ann->output + (ann->hidden_layers
                ? (ann->inputs + (ann->hidden) * (ann->hidden_layers-1))
                : 0);

        /* 更新每个输出神经元的权重 */
        for (j = 0; j < ann->outputs; ++j) {
            /* 更新偏置权重：Δbias = learning_rate * delta * (-1.0) */
            *w++ += *d * learning_rate * -1.0;
            /* 更新其他权重：Δweight = learning_rate * delta * input */
            for (k = 1; k < (ann->hidden_layers ? ann->hidden : ann->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }

        assert(w - ann->weight == ann->total_weights);
    }

    /**
     * 第五步：更新隐藏层的权重
     * 
     * 与更新输出层权重的原理相同，
     * 只是从最后一个隐藏层开始，逐层向前更新
     */
    for (h = ann->hidden_layers - 1; h >= 0; --h) {

        /* 定位当前隐藏层的delta */
        double const *d = ann->delta + (h * ann->hidden);

        /* 定位当前层的输入（上一层的输出） */
        double const *i = ann->output + (h
                ? (ann->inputs + ann->hidden * (h-1))
                : 0);

        /* 定位当前隐藏层的权重起始位置 */
        double *w = ann->weight + (h
                ? ((ann->inputs+1) * ann->hidden + (ann->hidden+1) * (ann->hidden) * (h-1))
                : 0);

        /* 更新每个隐藏神经元的权重 */
        for (j = 0; j < ann->hidden; ++j) {
            /* 更新偏置 */
            *w++ += *d * learning_rate * -1.0;
            /* 更新其他权重 */
            for (k = 1; k < (h == 0 ? ann->inputs : ann->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }
    }
}


/**
 * @brief 将神经网络保存到文件
 * @param ann 神经网络指针
 * @param out 已打开的输出文件指针
 * 
 * 保存的文件格式：
 * 第一行：inputs hidden_layers hidden outputs（四个整数）
 * 后续：total_weights 个浮点数（科学计数法，保留20位精度）
 * 
 * 使用示例：
 * @code
 * FILE *f = fopen("network.ann", "w");
 * genann_write(ann, f);
 * fclose(f);
 * @endcode
 * 
 * @note 使用 %.20e 格式确保浮点数精度不会丢失
 */
void genann_write(genann const *ann, FILE *out) {
    /* 先写入网络架构参数 */
    fprintf(out, "%d %d %d %d", ann->inputs, ann->hidden_layers, ann->hidden, ann->outputs);

    /* 再写入所有权重值（使用科学计数法，高精度） */
    int i;
    for (i = 0; i < ann->total_weights; ++i) {
        fprintf(out, " %.20e", ann->weight[i]);
    }
}


