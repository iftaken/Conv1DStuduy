from Conv1DCompute import Tree



if __name__ == '__main__':
    tree = Tree([1 for _ in range(100)])
    # 反卷积
    # MultiBand MelGan Model struct
    # https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_ckpt_0.1.1.zip
    tree.conv1d(kernel_size=7,  padding=3)
    tree.dcov1D(kernel_size=10, stride=5, padding=3, outpadding=1)
    tree.conv1d(kernel_size=3, dilation=1, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=9, padding=9)
    tree.conv1d(kernel_size=3, dilation=27, padding=27)
    
    tree.dcov1D(kernel_size=10, stride=5, padding=3, outpadding=1)
    tree.conv1d(kernel_size=3, dilation=1, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=9, padding=9)
    tree.conv1d(kernel_size=3, dilation=27, padding=27)
    
    tree.dcov1D(kernel_size=6, stride=3, padding=2, outpadding=1)
    tree.conv1d(kernel_size=3, dilation=1, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=9, padding=9)
    tree.conv1d(kernel_size=3, dilation=27, padding=27)
    
    # tree.conv1d(kernel_size=7, dilation=1, padding=3)
    
    # print(tree)

        
    tree.trace_back()
    # print(tree)
    effect_index = tree.count_layer_0_effect()
    # 回溯到最开始有哪些点被感染了
    print(effect_index)
    print(len(effect_index))
    left_pad, right_pad = tree.compute_pad(effect_index)
    print(f"通过overlap 消除计算的影响: 左边：{left_pad} 右边: {right_pad}")
    