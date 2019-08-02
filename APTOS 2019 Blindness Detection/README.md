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
