# EFF_B5 

import
add Codeadd Markdown
import random
import pandas as pd
import numpy as np
import os
import cv2
​
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
​
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
​
from tqdm.auto import tqdm
​
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
​
import torchvision.models as models
​
from sklearn.metrics import f1_score
​
import warnings
warnings.filterwarnings(action='ignore') 
add Codeadd Markdown
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device
device(type='cuda')
add Codeadd Markdown
하이퍼파라미터 세팅
add Codeadd Markdown
CFG = {
    'IMG_SIZE':224,
    'EPOCHS':20,
    'LEARNING_RATE':0.001,
    'BATCH_SIZE':16,
    'SEED':42
}
add Codeadd Markdown
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(CFG['SEED']) # Seed 고정
add Codeadd Markdown
데이터 전처리
add Codeadd Markdown
df = pd.read_csv('/kaggle/input/artist-data/train.csv')
df.loc[(df['id'] == 3896) & (df['artist'] == 'Titian'), ['img_path', 'id', 'artist']] = ['./train/3986.jpg', 3986, 'Alfred Sisley']
df.loc[(df['id'] == 3896) & (df['artist'] == 'Edgar Degas'), 'artist'] = 'Titian'
df.to_csv('/kaggle/working/new_train_data.csv', index=False)
df = pd.read_csv('/kaggle/working/new_train_data.csv')
df.head()
id	img_path	artist
0	0	./train/0000.jpg	Diego Velazquez
1	1	./train/0001.jpg	Vincent van Gogh
2	2	./train/0002.jpg	Claude Monet
3	3	./train/0003.jpg	Edgar Degas
4	4	./train/0004.jpg	Hieronymus Bosch
add Codeadd Markdown
encoder = preprocessing.LabelEncoder()
df['artist'] = encoder.fit_transform(df['artist'].values)
df.head(3)
id	img_path	artist
0	0	./train/0000.jpg	9
1	1	./train/0001.jpg	48
2	2	./train/0002.jpg	7
add Codeadd Markdown
train_df, val_df, _, _ = train_test_split(df, df['artist'], test_size=0.2,
                                          random_state=CFG['SEED'])
add Codeadd Markdown
train_df = train_df.sort_values(by=['id'])
val_df = val_df.sort_values(by=['id'])
​
display(train_df.head(3))
display(val_df.head(3))
id	img_path	artist
0	0	./train/0000.jpg	9
1	1	./train/0001.jpg	48
2	2	./train/0002.jpg	7
id	img_path	artist
8	8	./train/0008.jpg	31
12	12	./train/0012.jpg	11
14	14	./train/0014.jpg	15
add Codeadd Markdown
DataLoader
add Codeadd Markdown
def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values
add Codeadd Markdown
train_img_paths, train_labels = get_data(train_df) # 4728개
val_img_paths, val_labels = get_data(val_df) # 1183개
add Codeadd Markdown
temp = train_img_paths[0]
#/kaggle/input/artist-data/train
temp2 = temp.replace('./t', '/t')
'/kaggle/input/artist-data' + temp2
'/kaggle/input/artist-data/train/0000.jpg'
add Codeadd Markdown
나만의 데이터셋
add Codeadd Markdown
class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = img_path.replace('./t', '/t')
        img_path = ('/kaggle/input/artist-data' + img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
add Codeadd Markdown
train_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE']*2, CFG['IMG_SIZE']*2),
    A.RandomCrop(p=1, height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.5),
    A.OneOf([
        A.MotionBlur(p=1),
        A.OpticalDistortion(p=1),
        A.GaussNoise(p=1)
    ], p=0.3),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])
​
test_transform = A.Compose([
    A.Resize(CFG['IMG_SIZE']*2,CFG['IMG_SIZE']*2),
    A.RandomCrop(p=1, height=CFG['IMG_SIZE'], width=CFG['IMG_SIZE']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0, always_apply=False, p=1.0),
    ToTensorV2()
])
add Codeadd Markdown
train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)
​
val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
add Codeadd Markdown
모델 정의
add Codeadd Markdown
ResNet151
add Codeadd Markdown
# 모델 아키텍처 가져오기
model_ResNet151 = models.resnet152(pretrained=True)
​
# 모두 학습 불가로 만들기
for parameter in model_ResNet151.parameters():
    parameter.requires_grad = False
​
# 마지막 CNN과 classifier에 한하여 학습 가능으로 바꾸기
model_ResNet151.layer4.requires_grad = True
model_ResNet151.fc.requires_grad = True
​
# 분류기에서 50진분류로 만들기
model_ResNet151.fc # Linear(in_features=2048, out_features=1000, bias=True)
model_ResNet151.fc = nn.Linear(in_features=2048, out_features=50, bias=True)
add Codeadd Markdown
EfficienceNet
add Codeadd Markdown
class BaseModel(nn.Module):
    def __init__(self, num_classes=50):
        super(BaseModel, self).__init__()
        self.backbone = models.efficientnet_b5(pretrained=True)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=1000, out_features=num_classes),
        )
​
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x
add Codeadd Markdown
model_Eff = BaseModel(num_classes=50)
add Codeadd Markdown
훈련을 시켜봅시다!
add Codeadd Markdown
def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")
add Codeadd Markdown
def validation(model, criterion, test_loader, device):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    
    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            val_loss.append(loss.item())
            
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()
        
    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1
add Codeadd Markdown
def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    best_score = 0
    best_model = None
    
    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = []
        
        for img, label in tqdm(iter(train_loader)):
            img = img.float().to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            model_pred = model(img)
            
            loss = criterion(model_pred, label)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        tr_loss = np.mean(train_loss)
        
        val_loss, val_score = validation(model, criterion, test_loader, device)
        
        print(f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')
        
        if scheduler is not None:
            scheduler.step(metrics=val_score)
        
        if best_score < val_score:
            best_model = model
            best_score = val_score
    return best_model
add Codeadd Markdown
훈련해보기
add Codeadd Markdown
model_Eff.eval()
optimizer_Eff = torch.optim.Adam(params=model_Eff.parameters(), lr=CFG['LEARNING_RATE'])
scheduler_Eff = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer_Eff,
                                                 mode='max', factor=0.1,
                                                 patience=3, verbose=True)
add Codeadd Markdown
infer_model = train(model_Eff, optimizer_Eff, train_loader, val_loader,
                    scheduler_Eff, device=device)
100%
296/296 [02:43<00:00, 2.02it/s]
100%
74/74 [00:22<00:00, 3.46it/s]
Epoch [1], Train Loss : [2.87462] Val Loss : [2.64398] Val F1 Score : [0.18323]
100%
296/296 [02:43<00:00, 1.99it/s]
100%
74/74 [00:22<00:00, 3.38it/s]
Epoch [2], Train Loss : [2.30234] Val Loss : [2.44591] Val F1 Score : [0.26239]
100%
296/296 [02:43<00:00, 2.14it/s]
100%
74/74 [00:22<00:00, 3.47it/s]
Epoch [3], Train Loss : [2.02473] Val Loss : [2.04240] Val F1 Score : [0.32345]
100%
296/296 [02:43<00:00, 1.90it/s]
100%
74/74 [00:22<00:00, 3.42it/s]
Epoch [4], Train Loss : [1.81377] Val Loss : [1.89470] Val F1 Score : [0.38077]
100%
296/296 [02:42<00:00, 2.07it/s]
100%
74/74 [00:22<00:00, 3.46it/s]
Epoch [5], Train Loss : [1.68213] Val Loss : [1.95974] Val F1 Score : [0.38676]
100%
296/296 [02:42<00:00, 2.06it/s]
100%
74/74 [00:22<00:00, 3.41it/s]
Epoch [6], Train Loss : [1.59871] Val Loss : [1.71862] Val F1 Score : [0.44326]
100%
296/296 [02:42<00:00, 2.05it/s]
100%
74/74 [00:22<00:00, 3.07it/s]
Epoch [7], Train Loss : [1.47838] Val Loss : [1.65833] Val F1 Score : [0.46579]
100%
296/296 [02:44<00:00, 2.02it/s]
100%
74/74 [00:22<00:00, 2.61it/s]
Epoch [8], Train Loss : [1.43392] Val Loss : [1.59974] Val F1 Score : [0.47804]
100%
296/296 [02:44<00:00, 2.11it/s]
100%
74/74 [00:22<00:00, 3.12it/s]
Epoch [9], Train Loss : [1.32727] Val Loss : [1.58792] Val F1 Score : [0.49641]
100%
296/296 [02:44<00:00, 1.96it/s]
100%
74/74 [00:22<00:00, 3.08it/s]
Epoch [10], Train Loss : [1.27169] Val Loss : [1.75000] Val F1 Score : [0.49663]
100%
296/296 [02:45<00:00, 1.92it/s]
100%
74/74 [00:22<00:00, 3.30it/s]
Epoch [11], Train Loss : [1.22789] Val Loss : [2.05651] Val F1 Score : [0.46335]
100%
296/296 [02:45<00:00, 2.00it/s]
100%
74/74 [00:22<00:00, 3.39it/s]
Epoch [12], Train Loss : [1.19449] Val Loss : [1.97321] Val F1 Score : [0.51312]
100%
296/296 [02:43<00:00, 2.00it/s]
100%
74/74 [00:22<00:00, 3.18it/s]
Epoch [13], Train Loss : [1.14031] Val Loss : [1.53147] Val F1 Score : [0.55355]
100%
296/296 [02:45<00:00, 1.73it/s]
100%
74/74 [00:22<00:00, 3.37it/s]
Epoch [14], Train Loss : [1.08434] Val Loss : [2.02438] Val F1 Score : [0.47329]
100%
296/296 [02:45<00:00, 1.78it/s]
100%
74/74 [00:22<00:00, 3.42it/s]
Epoch [15], Train Loss : [1.05979] Val Loss : [1.79056] Val F1 Score : [0.50662]
100%
296/296 [02:46<00:00, 1.75it/s]
100%
74/74 [00:22<00:00, 3.45it/s]
Epoch [16], Train Loss : [1.04914] Val Loss : [1.62764] Val F1 Score : [0.54951]
55%
```

