# Get start with TVM
# 2021.7.1-7.6
官方：https://tvm.apache.org/docs/api/python/index.html

   (1):安装TVM开发环境，从官网和其他blog了解了TVM的编译以及优化的过程，以及优化的主要节点主要在哪部分做 。![enter image description here](https://raw.githubusercontent.com/tlc-pack/web-data/main/images/design/tvm_dyn_workflow.svg)


   (2):结合官方的文档，实现了MobileNetv2(pytorch)在CPU上的编译，从单张图片的时间推理来看，MobileNetv2直接在cpu上的推理时间约为17ms;而经过TVM编译后的MobileNetv2(转换成ONNX后编译的)在CPU上的同一张图片的推理时间为15ms,这里使用的是TVM的ContextPass的三级优化(共六级)，说明TVM在CPU上可以实现模型加速。

   (3):改变模型推理的平台，在GPU上重复上述过程，区别是直接用TVM编译pytorch模型，不转换成ONNX，时间上TVM编译后的推理时间比直接在GPU上的推理时间长(0.02s-->0.3s)，后面发现CPU和GPU的两个编译函数不一样，relay.build_module.create_executor()和relay.build()的区别，需要看一下这两者的区别。还有一个可能的原因是，我测试的是一张图片的推理时间，可能会存在一开始会有震荡，测多次取平均值可能更能说明问题。

   (4):针对这个问题，查找了一些资料，一个较大的可能是没有针对GPU做任何优化，导致直接编译后的模型推理效率很低，GitHub上发现别人做的实验也有类似情况，直接部署到Jetson Nano从CPU到GPU的时间变化为160ms-->1.8s.(https://mp.weixin.qq.com/s/7Wvv4VOPdj6N_CEg8bJFXw)


(5):为了探索(4)的问题，根据官网的文档，目前在用TVM的Autotune做一个自动优化，现在只针对2D_Conv这个Op,过程非常慢，目前还没搞定
 
调查慢的原因：
AutoTVM模块通过搜索的方式获得具体硬件下的最优调度配置，总体步骤如：

-   针对每个Relay子函数，定义一系列调度原语（Schedule primitives），通过调度原语的组合实现计算结果等价的Relay子函数
-   在海量的配置组合中为每个Relay子函数搜索最优的调度配置

对于卷积这样的计算密集算子，调度配置的搜索空间通常有几十亿种选择，比较棘手的问题就是如何高效的在搜索空间中获得最优/局部最优解。常见的做法有两种：

-   遍历所有可能的参数编译模型并评估
-   评估部分参数模型，训练一个性能评估函数，用于指导搜索过程寻找更好的调度配置

由于搜索空间过于巨大，第一种方法需要花费的时间难以估量；AutoTVM采用的是第二种，用一个XGBoost模型（其他回归预测模型也可）作为评估调度配置性能的模型（CostModel），用以指导模拟退火算法（默认）寻找最优调度配置。


# 2021.7.7
## model without tuning
https://github.com/leoluopy/autotvm_tutorial

在运行代码的时候，突然报错：
CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. 
解决办法
- export CUDA_VISIBLE_DEVICES = 0 (无效)
- 重装torch,reboot(OK) 

没有对pytorch模型进行转换，因为TVM支持直接编译pytorch模型，结果：

(1):单张图片在GPU的推理时间：33.50ms

(2):600张图片在GPU的平均推理时间：12.60ms

(3):TVM编译pytorch模型之后，推理测试600张的图片的平均时间：14.99ms (std: 0.47ms)

可以发现，这个时候没有对模型进行优化，推理时间是比原始的pytorch模型是更久的。

## autoTVM
使用TVM自带的自动调优的函数来进行优化，关键函数：autotvm.apply_history_best(log_file)。.log包括一些优化的配置，但是还没看明白做了哪些优化。
然后进行重新编译，依然使用PassContext()的三级优化来编译，TVM编译调优的pytorch模型后，推理测试600张的图片的平均时间：6.16ms (std: 0.30 ms)

|method| mean time(ms) |
| -- |--|
| pytorch+GPU | 12.60 | 
| TVM  |  14.99 |   |
|  TVM+autoTVM | 6.16 |  

  
优化VS不优化：Gain = 14.99ms - 6.16ms = 8.83ms

优化VS pytorch: Gain = 12.60ms - 6.16ms = 5.94ms

## 小结
- 基本解决了上一周的两个疑惑，直接使用一张图片来测试推理时间，前后会有一些波动，单张33.5ms vs 平均12.60ms
- 两次的实验MobileNetv2、CenterFace(上周GitHub的blog也说明了这个问题)验证了不经过优化的模型直接用TVM编译并没有加速的效果，反而时间变长了，因此在后期的模型优化过程，需要自动调优或者手动调优才能实现加速。

# 2021.7.8
## TVM int8 quantize
模型：
接口：mod  =  relay.quantize.quantize(mod, params)
最新的0.8版本官方文档没有找到这个接口....
考虑用pytorh 的量化包来实现，官方文档说目前还不支持用cuda做训练后静态量化，所以思路：pytorch model---->cpu quantization ------->TVM + gpu  ------->autoTVM
安装依赖：https://github.com/pytorch/FBGEMM ，支持低精度量化的运算
坑：
pytorch static quantization:不支持反卷积的量化, 不支持add等操作




<!--stackedit_data:
eyJoaXN0b3J5IjpbODEwMDc1MTA5LDExNzAyOTg2NzQsNDU0Mj
E1MzYzLC0xNDEzODA0Mzk0LDQ0MDQ4NzQxMCwtMTY4NDA3Nzk0
OCwtMTAxNDkxNDQ4MCwtNTc1MjIyNDQ2LDUwODE1ODE5MiwxOT
kxMjg4MzgsLTE1NjE5ODAyODIsLTMzMzkwNjgyNywtMjc1NjI3
NDQ5LDk2MTMyNjEyMV19
-->