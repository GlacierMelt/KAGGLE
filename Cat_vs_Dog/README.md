# Cat vs Dog

## Create submission.csv
```python
import pandas as pd
import os
submission_df = pd.DataFrame.from_dict({'id': os.listdir(f'./data/test'),
                                        'label': test_pred})

submission_df['id'] = submission_df['id'].map(lambda x: x.split('.')[0])
submission_df['id'] = submission_df['id'].astype(int)
submission_df = submission_df.sort_values('id')

submission_df.to_csv('submission.csv', index = False)
submission_df.head()
```
