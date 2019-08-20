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
---
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
#### 3. Ben&Crop Preprocessing
```python
import os
folder_name = 'Preprocess_image'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def Ben_Preprocessing(image):
    lower = np.array([7,7,7]) 
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)
    img_R = image[:,:,0][np.ix_(mask.any(1), mask.any(0))]
    img_G = image[:,:,1][np.ix_(mask.any(1), mask.any(0))]
    img_B = image[:,:,2][np.ix_(mask.any(1), mask.any(0))]
    image = np.stack([img_R, img_G, img_B], axis=-1)
    image = cv2.resize(image, (512,512))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 30) ,-4 ,128)
    return image
    
 def Ben_Crop_Saver(csv_file, image_dir, new_image_dir, image_type):
    data = pd.read_csv(csv_file)
    for idx in range(len(data)):
        img_name = os.path.join(image_dir, data.iloc[idx, 0] + image_type)
        image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
        image = Ben_Preprocessing(image)
        image = cv2.resize(image, (224,224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print(os.path.join(image_dir,img_name))
        cv2.imwrite(os.path.join(new_image_dir, data.iloc[idx, 0]+ '.png'), image)
    print('Number of images: ', len(os.listdir(new_image_dir)))
 ```
#### 4. Train Split Test
```python
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('./Data Preprocessing/train.csv')
train_df, val_df = train_test_split(train_data, test_size=0.05, random_state=46, stratify=train_data.diagnosis)
```
#### ADVERSARIAL EXAMPLE GENERATION
```python
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image
```
![image](https://github.com/GlacierMelt/KAGGLE/blob/master/APTOS%202019%20Blindness%20Detection/image/adversarial_image.PNG)
### Training
---
### Submission
---
#### TTA
```python
start = time.time()
model.cuda().eval()
prediction_R = np.zeros((num_image, 1))
with torch.no_grad():
    for _ in tqdm(range(TTA), desc='Inferencing'):
        prefetcher = data_prefetcher(test_loader, 'test')
        for i in range(len(test_loader)):
            images = prefetcher.next()
            outputs = model(images)
            predicted = outputs.detach().cpu().numpy().squeeze().reshape(-1, 1)
            prediction_R[i * batch_size:( i + 1) * batch_size] += predicted
prediction_R = prediction_R / TTA
prediction_R = prediction(prediction_R).astype(int)
print(emoji.emojize("Time: %.5fmin 🍹" % ((time.time()-start)/60.0)))
```
#### ADD CSV
```python
submission_df = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
submission_df['diagnosis'] = prediction_R
submission_df.to_csv('submission.csv', index = False)
submission_df.head(10)
```
