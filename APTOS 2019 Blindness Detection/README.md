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

**Data Preprocessing**
##### 1.Downlod dataset
```PYTHON
!apt-get -y update && apt-get install -y zip
!apt-get -y update && apt-get install -y wget
!wget 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/14774/536888/all.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1565001228&Signature=ecTeTu%2F9gIQKcEJePgTnnI5G5hoesl3zOgN%2BV3cILfJxnlsZgjp5f2CK1yMG2vZi2VrVxRAbEXiOE903Uw10B1bxpHs%2Bjesu4R8VaXYjYleB1wAQcLWpnUx2Kvki9G6R6C9nOxH178%2FTHz4Hbl%2FpxolKWgRCoNECc%2BXvFTSTAAL5TxHleJIekIecROR7Rid2N1KXG%2FrGrTcKOhgakdmb2gilJpxMSSa1beUrDEJ3E6aHv8X3gFtsxsdTm8hI087U0kuhl9oNiHhNCxtTcl%2BPUGj9POUkOe%2FqCiZo95IEouXC%2FR57kLWtrpQ9846Y144vCquC8pEi%2BX0GGwwqL1sQ5g%3D%3D&response-content-disposition=attachment%3B+filename%3Daptos2019-blindness-detection.zip' --no-check-certificate -O aptos2019-blindness-detection.zip.zip
```
