import glob
import cv2
import pandas as pd

def make_df(train_dir='./train.csv', test_dir='./test.csv'):
    ```
    경로를 지정해주면 train_labels 데이터프레임과
    test_labels 데이터프레임을 반환합니다.
    ```
    train_labels = pd.read_csv(train_dir, index_col=1)
    test_labels = pd.read_csv(test_dir, index_col=1)
    
    return train_labels, test_labels


def label_encoder(train_labels=None):
    ```
    train_labels 데이터프레임에 대해
    Label Encoding이 적용된 컬럼을 새로 생성하여 반환합니다.
    ```
    train_labels = train_labels.copy()
    train_labels['LE_artist'] = train_labels['artist'].astype('category').cat.codes

    return train_labels


def get_labelmap(train_labels=None):
    ```
    Label Encoding 전후를 매치하여
    Dictionary형태로 반환합니다.
    ```
    label_map = dict(enumerate(train_labels['artist'].astype('category').cat.categorise))
    return label_map


class PaintingsDataset():
    def __init__(self, root_dir, labels_df=None,  transform=None):
        self.filepaths = sorted(glob.glob(root_dir + '*.jpg'))
        self.transform = transform
        if labels_df is not None: # train일때
            self.labels_df = labels_df
            
        else: self.labels_df = None # test일때
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        # (1) 이미지 준비
        image_filepath = self.filepaths[idx]

        image = cv2.imread(image_filepath) # np.array형태(BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # np.array형태(RGB)로 반환
                
        # (2) Label 준비
        if self.labels_df is not None: # train일때
            file_name = image_filepath
            painter = self.labels_df.loc['./'+file_name]['LE_artist']
        
        # (3) 반환할 이미지와 Label 만들기
        
        # 딕셔너리화
        if self.labels_df is not None: # csv가 들어왔을때 (train)
            imagelabel_dict1 = self.transform(image=image)
            imagelabel_dict1['label'] = painter
            return imagelabel_dict1
        
        else: # csv가 들어오지 않았을때 (test)
            imagelabel_dict2 = self.transform(image=image)
            return imagelabel_dict2
        