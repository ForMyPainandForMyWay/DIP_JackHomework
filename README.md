数字图像处理-结课作业

基于opencv的数字图像滤波

**项目环境：**

+ C++14
+ OpenCV 4.9.0
+ VS2022

**简述：**

+ 真实豪吃的一次大作业啊。战斗，爽！
+ 遇到的问题包括但不限于：
	1. Mat元素进行数值类型转换时出现数值移除问题（我操你daddy坑死我了😡）
	2. 多线程函数传参以及数据同步问题，详细来讲是：利用ref函数实现左值引用传递，避免创建thread对象时将引用进行拷贝；需要使用move转移promise的语义(使用ref传递引用会不通过编译)等等
+ 待优化的部分：彻底地并行化，对于像素的遍历尚未完成并行操作，Mat和CUDA通信只能直接把内存数据拷过去吗？？？