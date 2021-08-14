import numpy as np
aa = [1, 2, 3]

# 왼쪽 2개, 오른쪽 3개 0으로 패딩합니다
print(np.pad(aa, (2, 3), 'constant', constant_values=0))
