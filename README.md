origin ：https://github.com/1ytic/warp-rnnt

# 问题：
对比过paddle(aistudio)和pytorch(colab)的输入和结果：

1、输入都是采取numpy转tensor，并且随机数种子一样，打印出来输入的tensor数字是一模一样的

2、运算过程在core.cu中实现，.cu文件paddle和pytorch是一样的

3、问题出现在[paddle](./paddle/core.cu#353)上面(对应于[pytorch](./pytorch/core.cu#353))，这一步应该是手算出来的结果和机器算出来的结果的比较，pytorch算出来相差不大，但是paddle算出来的有差异，不知道问题出现在哪里

# Paddle bindings for CUDA-Warp RNN-Transducer

## Install
under paddle folder
```bash
python setup.py install
```

## Test
under paddle folder
```bash
python test.py
```

# Pytorch bindings for CUDA-Warp RNN-Transducer

## Install
under pytorch folder
```bash
python setup.py install
```

## Test
under pytorch folder
```bash
python test.py
```
