# Conv1D 与 Conv1D_Transpose学习笔记

可视化验证 Conv1d 与 Conv1d_transpose 在组合和过程中，感受野的变化，推算出可以通过多少pad来消除边界计算的影响。

## 计算验证

```shell

# 基础计算验证
python Conv1DCompute.py

# 验证 melGan 模型的
python demo_melgan.py

# 验证 hifiGan 模型

python demo_hifigan.py


```

