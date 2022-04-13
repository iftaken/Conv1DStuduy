from turtle import pos
import matplotlib.pyplot as plt
from Conv1DCompute import Tree

# r = 0.01
# circle1 = plt.Circle((0, 0), r, color='g')
# circle3 = plt.Circle((1, 1), r, color='r')

# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot

# ax.add_patch(circle1)
# ax.add_patch(circle3)
# ax.arrow(0+r, 0+r, 1-3*r, 1-3*r, head_width=r)

# fig.savefig('plotcircles.png')

def add_node(ax, node, position:tuple, r=0.01, first_node=False):
    if node.is_effect():
        if first_node:
            circle = plt.Circle(position, r, color='b')
        else:
            circle = plt.Circle(position, r, color='r')
    else:
        circle = plt.Circle(position, r, color='g')
    ax.add_patch(circle)
    # return ax


def add_arrow(ax, start_position, end_position, r=0.01):
    ax.arrow(start_position[0]+r, start_position[1]+r, end_position[0]-3*r, end_position[1]-3*r, head_width=r)


def compute_size(tree, width=1, height=1, sub_width=0.1, sub_height=0.05):
    # width = len(tree) * sub_width
    # height = max([len(n) for n in tree]) * sub_height
    
    sub_height = height / max([len(n) for n in tree])
    sub_width = width / len(tree)
    
    plt.figure(figsize=(width, height))
    # plt.xlim(0, 10)
    # plt.ylim(0, 5)
    fig, ax = plt.subplots()
    return fig, ax, sub_width, sub_height

def compute_position(tree, height, sub_width=0.1, sub_height=0.05):
    postion_dict = {}
    
    for i, nodes in enumerate(tree):
        center_width = i * sub_width + sub_width / 2 
        center_height = height / 2
        if len(nodes) % 2 == 0:
            # 偶数
            center_index = len(nodes) // 2
            # 算每一个点的位置
            for j, node in enumerate(nodes):
                _width = center_width
                _height = center_height + (j - center_index + 0.5) * sub_height
                if str(node.layer) not in postion_dict:
                    postion_dict[str(node.layer)] = {}
                postion_dict[str(node.layer)][str(node.index)] = (_width, _height)   
        else:
            # 奇数
            center_index = (len(nodes) + 1) // 2
            for j, node in enumerate(nodes):
                _width = center_width
                _height = center_height - (j - center_index) * sub_height
                if str(node.layer) not in postion_dict:
                    postion_dict[str(node.layer)] = {}
                postion_dict[str(node.layer)][str(node.index)] = (_width, _height)
    
    return postion_dict

def plot_tree(tree, width=30, height=30, r=0.02, fig_name="Tree.png"):
    fig, ax, sub_width, sub_height = compute_size(tree, width=width, height=height)
    postion_dict = compute_position(tree, height, sub_width=sub_width, sub_height=sub_height)

    # 画 node
    for i, nodes in enumerate(tree):
        for node in nodes:
            position=postion_dict[str(node.layer)][str(node.index)]
            # print(position)
            if i == 0:
                add_node(ax, node, position=position, r=r, first_node=True)
            else:
                add_node(ax, node, position=position, r=r, first_node=False)
    # ax.arrow(0+r, 0+r, 1-3*r, 1-3*r, head_width=r)
    # fig.savefig(fig_name,dpi=600,format='eps')
    # 画连线
    # for i, nodes in enumerate(tree):
    #     for node in nodes:
    #         if node.fathers:
    #             for father in node.fathers:
    #                 start_position = postion_dict[str(father.layer)][str(father.index)]
    #                 end_position = postion_dict[str(node.layer)][str(node.index)]
    #                 ax.arrow(start_position[0], start_position[1], end_position[0], end_position[1])
    #                 break
    #     break
    circle = plt.Circle((0,0), r, color='w')
    ax.add_patch(circle)
    ax.axis("equal")
    
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    fig.savefig(fig_name,dpi=2400)

            
if __name__ == '__main__':
    tree = Tree([1 for _ in range(30)])
    # 反卷积
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
    
    tree.conv1d(kernel_size=7, dilation=1, padding=3)
    
    
    tree.trace_back()
    # print(tree)
    effect_index = tree.count_layer_0_effect()
    # 回溯到最开始有哪些点被感染了
    print(effect_index)
    
    # 可视化展示
    plot_tree(tree.tree, width=600, height=600, r=0.02, fig_name="Tree.png")
    


            




