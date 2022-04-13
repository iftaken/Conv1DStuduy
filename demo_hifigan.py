from Conv1DCompute import Tree



if __name__ == '__main__':
    tree = Tree([1 for _ in range(100)])
    # 反卷积
    # HiFiGANGenerator
    tree.conv1d(kernel_size=7,  padding=3)
    # upsamples
    tree.dcov1D(kernel_size=10, stride=5, padding=3, outpadding=1)
    tree.dcov1D(kernel_size=10, stride=5, padding=3, outpadding=1)
    tree.dcov1D(kernel_size=8, stride=4, padding=2, outpadding=0)
    tree.dcov1D(kernel_size=6, stride=3, padding=2, outpadding=1)
    
    # HiFiGANResidualBlock 1
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=5, padding=5)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    
    # HiFiGANResidualBlock 2
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, dilation=3, padding=9)
    tree.conv1d(kernel_size=7, dilation=5, padding=15)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    
    # HiFiGANResidualBlock 3
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, dilation=3, padding=15)
    tree.conv1d(kernel_size=11, dilation=5, padding=25)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    
    # HiFiGANResidualBlock 4
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=5, padding=5)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    
    # HiFiGANResidualBlock 5
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, dilation=3, padding=9)
    tree.conv1d(kernel_size=7, dilation=5, padding=15)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    
    # HiFiGANResidualBlock 6
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, dilation=3, padding=15)
    tree.conv1d(kernel_size=11, dilation=5, padding=25)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    
    # HiFiGANResidualBlock 7
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=5, padding=5)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    
    # HiFiGANResidualBlock 8
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, dilation=3, padding=9)
    tree.conv1d(kernel_size=7, dilation=5, padding=15)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    
    # HiFiGANResidualBlock 9
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, dilation=3, padding=15)
    tree.conv1d(kernel_size=11, dilation=5, padding=25)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    
    # HiFiGANResidualBlock 10
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, dilation=3, padding=3)
    tree.conv1d(kernel_size=3, dilation=5, padding=5)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    tree.conv1d(kernel_size=3, padding=1)
    
    # HiFiGANResidualBlock 11
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, dilation=3, padding=9)
    tree.conv1d(kernel_size=7, dilation=5, padding=15)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    tree.conv1d(kernel_size=7, padding=3)
    
    # HiFiGANResidualBlock 12
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, dilation=3, padding=15)
    tree.conv1d(kernel_size=11, dilation=5, padding=25)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)
    tree.conv1d(kernel_size=11, padding=5)

    # output
    tree.conv1d(kernel_size=7, padding=3)
    
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
    