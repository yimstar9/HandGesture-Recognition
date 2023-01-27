#뎁스 152 20회 배치2 0.952475228475228
#뎁스 50 50회 배치2 0.9741382007822684
#뎁스 50 100회 배치8 0.9608283608283609
import random
import pandas as pd
import numpy as np
import os
import cv2
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from functools import partial


from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import warnings
from glob import glob #폴더내 파일 리스트 불러오기

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


os.chdir("E:\GoogleDrive\pycv\리모콘 제스쳐")
os.getcwd()
import math

from torch.optim.lr_scheduler import _LRScheduler

#'먼저 warm up을 위하여 optimizer에 입력되는 learning rate = 0 또는 0에 가까운 아주 작은 값을 입력합니다.
# 위 코드의 스케쥴러에서는 T_0, T_mult, eta_max 외에 T_up, gamma 값을 가집니다.
# T_0, T_mult의 사용법은 pytorch 공식 CosineAnnealingWarmUpRestarts와 동일합니다. eta_max는 learning rate의 최댓값을
# 뜻합니다. T_up은 Warm up 시 필요한 epoch 수를 지정하며 일반적으로 짧은 epoch 수를 지정합니다. gamma는 주기가 반복될수록 eta_max
# 곱해지는 스케일값 입니다.'
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

CFG = {
    'FPS':30,
    'IMG_SIZE':128,
    'EPOCHS':100,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':8,
    'SEED':41
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG['SEED']) # Seed 고정


df = pd.read_csv('./train_jpg.csv')
test = pd.read_csv('./test_jpg.csv')
df2 = pd.read_csv('./train_jpg2.csv')
# test2 = pd.read_csv('./test_jpg2.csv')

train_data, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CFG['SEED'])
train2_data, val2, _, _ = train_test_split(df2, df2['label'], test_size=0.2, random_state=CFG['SEED'])

train_data = pd.concat([train_data,train2_data],ignore_index=True)
val_data = pd.concat([val,val2],ignore_index=True)

class CustomDataset(Dataset):
    def __init__(self, video_path_list, label_list):
        self.video_path_list = video_path_list
        self.label_list = label_list

        #self.video_id = video_id
    def __getitem__(self, index):
        frames = self.get_video(self.video_path_list[index])

        if self.label_list is not None:
            label = self.label_list[index]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_path_list)

    def get_video(self, path):
        frames = []
        folder_lists = glob(path + '\**')
        file_list = [f'frame_{i:02d}.jpg' for i in range(len(folder_lists))]
        for j in range(len(file_list)):
            image = cv2.imread(folder_lists[j])
            image = image / 255.
            frames.append(image)
        return torch.FloatTensor(np.array(frames)).permute(3,0, 1, 2)
#
# mean = [0.45, 0.45, 0.45]
# std = [0.250, 0.250, 0.250]
#

train_dataset = CustomDataset(train_data['path'].values, train_data['label'].values)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

# train_dataset.__getitem__(0)[0].shape
val_dataset = CustomDataset(val_data['path'].values, val_data['label'].values)
val_loader = DataLoader(val_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

#병목
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  ##다운샘플 적용

        out += residual ## 아이덴티티 매핑 적용
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, block_inplanes, n_input_channels=3, conv1_t_size=7, conv1_t_stride=1,
                 no_max_pool=False, shortcut_type='B', widen_factor=1.0, n_classes=5):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    running_loss = 0
    correct = 0
    total = 0

    best_val_score = 0
    best_model = None
    best_epoch =0
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        for videos, labels in tqdm(iter(train_loader)):

            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(videos)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # 텐서보드 그래프용 통계
            _, predicted = output.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

        accu = 100. * correct / total

        _val_loss, _val_score = validation(model, criterion, val_loader, device,epoch)
        _train_loss = np.mean(train_loss)
        print(f'\nEpoch[{epoch}],TrainLoss:[{_train_loss:.3f}] ValLoss:[{_val_loss:.3f}] ValF1:[{_val_score:.3f}]')

        if scheduler is not None:
            scheduler.step(_val_score)

        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model
            best_epoch = epoch
        # 텐서보드 스칼라 작성
        writer.add_scalar('train loss', _train_loss,epoch)
        writer.add_scalar('train acc', accu,epoch)
    #텐서보드 모델구조
    writer.add_graph(model, videos)

    print(f'best epoch:{best_epoch}')
    return best_model


def validation(model, criterion, val_loader, device,epoch):
    model.eval()
    val_loss = []
    preds, trues = [], []
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in tqdm(iter(val_loader)):
            videos = videos.to(device)
            labels = labels.to(device)

            logit = model(videos)

            loss = criterion(logit, labels)

            val_loss.append(loss.item())

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()
            #텐서보드 통계
            running_loss += loss.item()
            _, predicted = logit.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        _val_loss = np.mean(val_loss)
    #텐서보드
    testloss = running_loss / len(val_loader)
    accu = 100. * correct / total ################통계
    writer.add_scalar('val loss', testloss,epoch)
    writer.add_scalar('val acc', accu,epoch)

    _val_score = f1_score(trues, preds, average='macro')
    return _val_loss, _val_score

model_depth = 50

kwargs  =  {'n_input_channels' : 3,
        'conv1_t_size' : 7,
        'conv1_t_stride' : 1,
        'no_max_pool' : False,
        'shortcut_type' : 'B',
        'widen_factor' : 1.0,
        'n_classes' : 5}
if model_depth == 10:
    model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
elif model_depth == 18:
    model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
elif model_depth == 34:
    model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 50:
    model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
elif model_depth == 101:
    model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
elif model_depth == 152:
    model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
elif model_depth == 200:
    model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

#model=torch.load('./Resnet3D ver3_model.pt')
model.eval()

# optimizer = torch.optim.Adam(model.parameters(), lr=CFG["LEARNING_RATE"])
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["LEARNING_RATE"] ,weight_decay=0.0001)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.8)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=10,T_mult=1, eta_max=0.1,  T_up=5, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10,T_mult=2, eta_min=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)
infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)

# torch.save(model, f'./Resnet3D 34 ver3_model.pt')

test_dataset = CustomDataset(test['path'].values, None)
test_loader = DataLoader(test_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=False, num_workers=0)


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for videos in tqdm(iter(test_loader)):
            videos = videos.to(device)

            logit = model(videos)

            preds += logit.argmax(1).detach().cpu().numpy().tolist()
    return preds

preds = inference(model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')

submit['label'] = preds
submit.head()

submit.to_csv('./Resnet_submit.csv', index=False)

T= pd.read_csv('./answer.csv')
print(f1_score(list(T.label), preds, average='macro'))
writer.close()
# tensorboard --logdir=runs