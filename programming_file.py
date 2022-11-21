## __init__()에서 사용할 네트워크 모델 정의
## forward()함수에서 모델에서 실행되어야할 계산을 좀 더 가독성 있게 코드로 작성

import torch.nn as nn
class MLP(nn.Module):
    def __int__(self):
        super(MLP,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True))

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x
model = MLP() ##모델에 대한 객체 생성

print("Printing children\n------------------")
print(list(model.children())) ##model.children()##같은 레벨의 하위 노드 반환
print("\n\nPrinting Modules\n------------------")
print(list(model.modules())) ##model.modules() ##모델의 네트워크에 대한 노드 반환

##nn.Sequential 모델의 계층이 복잡할수록 효과 높음