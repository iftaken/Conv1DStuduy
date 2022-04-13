class Node(object):
    def __init__(self, index, layer, value) -> None:
        self.value = value
        self.layer = layer
        self.index = index
        self.fathers = []
        self.sons = []
        self.next = None
        self.effective = 0
        self.effective_by_son = 0
    
    def __str__(self) -> str:
        return f"{self.layer}_{self.index}_{self.value}"
    
    def effect(self):
        # 表示被感染，感染不可逆
        self.effective = 1
    
    def effect_by_son(self):
        self.effective_by_son = 1
    
    def is_effect(self):
        return self.effective
    
    def setIndex(self, index):
        self.index = index
    
    def setValue(self, value):
        self.value = value
    
    def setNext(self, nextNode):
        self.next = nextNode
    
    def appendFather(self, father):
        self.fathers.append(father)
    
    def appendSon(self, son):
        self.sons.append(son)
    
    