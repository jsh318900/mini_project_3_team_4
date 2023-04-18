# EFF_B5 
- batch_size = 16
- lr = 0.001
```
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
```
```
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

