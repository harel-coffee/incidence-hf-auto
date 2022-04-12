import h5py

from deps.common import get_data_cached
from deps.pipelines import get_preprocessing
from hcve_lib.splitting import get_lco_splits
from hcve_lib.utils import loc
from hcve_lib.wrapped_sklearn import DFPipeline

data, metadata, X, y = get_data_cached()

splits = get_lco_splits(X, data)

preprocessing_steps, _, _ = get_preprocessing(X, standard_scaler=True)
transform_pipeline = DFPipeline(preprocessing_steps)

y_train = loc(splits['FLEMENGHO'][0], y)
y_test = loc(splits['FLEMENGHO'][1], y)

X_train = X.loc[splits['FLEMENGHO'][0]]
X_train = transform_pipeline.fit_transform(X_train, y_train)
X_train = X_train.astype('float32')

X_test = X.loc[splits['FLEMENGHO'][1]]
X_test = transform_pipeline.transform(X_test)
X_test = X_test.astype('float32')

file_write = h5py.File('./data/input.h5', 'w')

file_write.create_dataset('train/x', data=X_train)
file_write.create_dataset('train/e', data=y_train['data']['tte'])
file_write.create_dataset('train/t', data=y_train['data']['label'])

file_write.create_dataset('test/x', data=X_test)
file_write.create_dataset('test/e', data=y_test['data']['tte'])
file_write.create_dataset('test/t', data=y_test['data']['label'])
