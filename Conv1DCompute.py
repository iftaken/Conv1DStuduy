from node import Node
from typing import List

class Tree(object):
    def __init__(self, input_tensor:list) -> None:
        # tesor to node
        self.tree = []
        self.layer_names = []
        self.paddings = {}    # 记录 pad 所在的层以及对应的pad大小
        self.tensor2node(input_tensor)
        pass
    
    def __str__(self) -> str:
        
        # 逐层打印
        string = ""
        for layer_all, nodes in enumerate(self.tree):
            
            temps = []
            for node in nodes:
                temp = ""
                if node.is_effect():
                    temp += str(node.value) + "*"
                else:
                    temp += str(node.value)
                temps.append(temp)
            
                
            string += f"layer {nodes[0].layer}({self.layer_names[layer_all]}):\t{ ' | '.join(temps)} \n"
        return string
        
    
    def tensor2node(self, input_tensor):
        nodes = []
        layer = 0
        # tree 初始化第一层
        for index, value in enumerate(input_tensor):
            n = Node(index=index, layer=layer, value=value)
            if len(nodes) > 0:
                # 记录顺序关系
                nodes[-1].setNext(n)
            nodes.append(n)
            
        self.tree.append(nodes)
        self.layer_names.append("start")
    
    def conv1d(self, kernel_size:int, stride=1, dilation=1, padding=0):
        
        weight = 1
        bias = 0
        # 取当前层
        nodes = self.tree[-1]
        self.paddingNodes(nodes, padding)
        nodes = self.tree[-1]
        next_nodes = []
        # 先pad
        length = len(nodes)
        # 不变层
        layer = nodes[0].layer + 1
        
        # 通过 dilation 计算
        new_kernel_size = dilation * (kernel_size - 1) + 1
        # dilation - 1 表示空洞的大小
        
        next_index = 0
        for j in range( (length - new_kernel_size + 1) // stride ):
            start = stride * j
            # 把点对应关系找清楚
            fathers = []
            value = 0
            is_effect = False
            for i in range(kernel_size):
                now = start + i * dilation
                fathers.append(nodes[now])
                # 这里 用 weight =1 , bias 为 0 作为演示
                value += nodes[now].value * weight + bias
                if nodes[now].is_effect():
                    # 父结点中存在感染者
                    is_effect = True
            
            # 计算出的新点
            n = Node(index=next_index, layer=layer, value=value)
            n.fathers.extend(fathers)
            if is_effect:
                # 父节点
                n.effect()
            next_nodes.append(n)
            next_index += 1
        
        # 计算完成
        self.tree.append(next_nodes)
        # self.tree[-1] = next_nodes 
        self.layer_names.append("conv1d")
    
    def dcov1D(self, kernel_size=1, stride=1, padding=0, outpadding=0, dilation=1):
        # 暂时先不算 dilation
        # 反卷积
        weight = 1
        bias = 0
        nodes = self.tree[-1]
        new_nodes = []
        layer = nodes[0].layer + 1
        
        # 反卷积计算
        # kernel size & stride
        start = 0
        for i, node in enumerate(nodes):
            start = i * stride
            father = node
            for j in range(kernel_size):
                if len(new_nodes) > start + j:
                    # weight = 1, bias = 0
                    new_nodes[start + j].value += node.value * weight + bias
                    new_nodes[start + j].fathers.append(father)
                    if father.is_effect():
                        new_nodes[start + j].effect()
                    
                else:
                    n = Node(index=start + j-padding, layer=layer, value= node.value * weight + bias )
                    n.fathers.append(father)
                    if father.is_effect():
                        n.effect()
                    # else:
                    #     # father 没感染的情况下， 看 father 左侧被感染了
                    #     if father.index >0 and nodes[father.index - 1].is_effect():
                    #         n.effect()
                    #     # father 没有感染，但是 father 右侧被感染了
                    #     elif father.index < len(nodes) - 1 and nodes[father.index + 1].is_effect():
                    #         n.effect()
                    new_nodes.append(n)
        
        # 看看开头和结尾受影响的点
        # 会受到旁边影响的点
        new_length = len(new_nodes)
        effect_by_side = kernel_size - stride
        # left
        for i in range(effect_by_side):
            new_nodes[i].effect()
        # right
        for i in range(effect_by_side):
            new_nodes[new_length - i - 1].effect()
        
        # 处理 padding 与 outpadding
        new_nodes = new_nodes[padding: -(padding - outpadding)]
        # for i in range(len(new_nodes)):
        #     new_nodes[i].setIndex(i)
        self.layer_names.append("dconv1d")
        self.tree.append(new_nodes) 
        # self.tree[-1] = new_nodes 
          
    
    def paddingNodes(self, nodes:List[Node], padding=0, pad=0):
        layer = nodes[0].layer + 1
        self.paddings[str(layer)] = padding
        new_nodes = []
        if padding == 0:
            self.tree.append(nodes)
            self.layer_names.append("pad")
            return nodes
        else:
            # 前后各加padding大小的点
            # 开头pad
            for i in range(padding):
                n = Node(index=i, layer=layer, value=pad)
                # 这几个点是感染点
                n.effect()
                new_nodes.append(n)
            
            # 中间添加
            for i, node in enumerate(nodes):
                # n.setIndex(n.index + padding)
                n = Node(index=i+padding, layer=layer, value=node.value)
                if node.is_effect():
                    n.effect()
                new_nodes.append(n)
                new_nodes[-1].fathers.append(node)

            # 结尾pad
            for i in range(padding):
                n = Node(index=new_nodes[-1].index+1, layer=layer, value=pad)
                n.effect()
                new_nodes.append(n)
        self.tree.append(new_nodes)
        self.layer_names.append("pad")
        return new_nodes

    def count_layer_0_effect(self):
        # 统计输入受影响的情况
        nodes = self.tree[0]
        effect_indexs = [node.index for node in nodes if node.is_effect()]
        return effect_indexs
        

    def trace_back(self):
        # 从最下面一层网上进行回溯
        # 只有经过 transpose 层才需要找father
        # 同 conv1d层 向上对应感染
        depth = len(self.tree) - 1
        while depth > 0:
            nodes = self.tree[depth]
            for node in nodes:
                # 逐层标记
                self.check_father(node)
            depth -= 1
            
    
    def check_father(self, node):        
        if len(node.fathers) > 0:
            # 检查当前node是否感染
            if node.is_effect():
                for father in node.fathers:
                    now_layer = self.layer_names[node.layer]
                    if now_layer == "conv1d":
                        # father层一定是pad
                        
                        if father.index - self.paddings[str(father.layer)] == node.index:
                            self.tree[father.layer][father.index].effect()
                            # 一个结点只感染同index
                            break
                    elif now_layer == "pad":
                        # 向上传递
                        if father.index == node.index - self.paddings[str(node.layer)]:
                            self.tree[father.layer][father.index].effect()
                            break
                    elif now_layer == "dconv1d":
                        # 感染对应父结点
                        self.tree[father.layer][father.index].effect()
                    else:
                        # start
                        return None 
            else:
                # 对于 dconv1d来说，需要检查其父结点中是否存在感染者，如果存在，则
                return None
        else:
            return None
            
    def compute_pad(self, effect_index):
        left_pad = 0
        right_pad = 0
        
        for i in range(len(effect_index) - 1):
            diff = effect_index[i+1] - effect_index[i]
            if diff > 1:
                left_pad  = i + 1
                break
        
        right_pad = len(effect_index) - left_pad
        return left_pad, right_pad 
        
                    
if __name__ == '__main__':
    tree = Tree([1 for _ in range(40)])

    tree.conv1d(kernel_size=7,  padding=3)
    tree.dcov1D(kernel_size=10, stride=5, padding=3, outpadding=1)
    
    print(tree)
        
    tree.trace_back()
    print("##############")
    print(tree)
    effect_index = tree.count_layer_0_effect()
    # 回溯到最开始有哪些点被感染了
    print(effect_index)
    print(len(effect_index))
    print(tree.compute_pad(effect_index))
    