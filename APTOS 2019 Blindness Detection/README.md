# [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
**Checking CPU and GPU**
```PYTHON
import os
import multiprocessing
!nvidia-smi
mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
mem_gib = mem_bytes/(1024.**3)  
print("RAM: %f GB" % mem_gib)
print("CORES: %d" % multiprocessing.cpu_count())
```
**Data Sources**
* [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection/data)
* [2015 Blindness Detection](https://www.kaggle.com/tanlikesmath/diabetic-retinopathy-resized)
* [Diabetic Retinopathy: Segmentation and Grading Challenge](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)

### Data Preprocessing
#### 1. Download Dataset
```PYTHON
!apt-get -y update && apt-get install -y zip
!apt-get -y update && apt-get install -y wget
!wget 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/14774/536888/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1565001228&Signature=ecTeTu%2F9gIQKcEJePgTnnI5G5hoesl3zOgN%2BV3cILfJxnlsZgjp5f2CK1yMG2vZi2VrVxRAbEXiOE903Uw10B1bxpHs%2Bjesu4R8VaXYjYleB1wAQcLWpnUx2Kvki9G6R6C9nOxH178%2FTHz4Hbl%2FpxolKWgRCoNECc%2BXvFTSTAAL5TxHleJIekIecROR7Rid2N1KXG%2FrGrTcKOhgakdmb2gilJpxMSSa1beUrDEJ3E6aHv8X3gFtsxsdTm8hI087U0kuhl9oNiHhNCxtTcl%2BPUGj9POUkOe%2FqCiZo95IEouXC%2FR57kLWtrpQ9846Y144vCquC8pEi%2BX0GGwwqL1sQ5g%3D%3D&response-content-disposition=attachment%3B+filename%3Daptos2019-blindness-detection.zip' --no-check-certificate -O aptos2019-blindness-detection.zip
```
#### 2. Discarded  Duplicate Images  
```PYTHON
import imagehash
from keras.preprocessing.image import load_img
from multiprocessing import Pool

def create_hash(x):
    path = './Data Preprocessing/train_images/' + x + '.png'
    image = load_img(path) # PIL
    image_hash = str(imagehash.whash(image))
    return image_hash

def select_discard(df_train):
    data = df_train.copy()
    train_input1 = df_train[['img_hash']]
    train_input1['New']=1
    train_input2 = train_input1.groupby('img_hash').count().reset_index()
    train_input2 = train_input2[train_input2['New']>1]
    for idx in range(train_input2.shape[0]):
        df = df_train[df_train['img_hash']==train_input2.iloc[idx,0]]
        for i in range(5):
            if df[df['diagnosis']==i].shape[0] == 1:
                data.drop(df[df['diagnosis']==i].index, inplace=True)
    return data
```
