import logging
import sys
from abc import ABC
from typing import List, Dict, Union
from itertools import product
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:\t%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PredictionModel(ABC):
    def __init__(self, num_models:int=10, lag_value:int=3, data_dir:str='data/', weight_dir:str='weight/') -> None:
        self.lag_frame = [i+1 for i in range(lag_value)]
        self.index_cols = ['date_block_num', 'shop_id', 'item_id']
        self.model_lst = self.init_models(num_models, weight_dir)
        self.train_data = pd.read_csv(data_dir + 'train.csv', index_col=0)
        self.test_data = pd.read_csv(data_dir + 'test.csv')
        self.items_df = pd.read_csv(data_dir + 'items.csv')
        self.item_cat_df = pd.read_csv(data_dir + 'item_categories.csv')
        self.shops_df = pd.read_csv(data_dir + 'shops.csv')
        self.processed_data = self._data_preprocessing()
        logger.info("Done preprocessing dataset")
        

    def init_models(self, num_models:int, weight_dir:str='weight/'):
        model_lst = []
        for i in range(num_models):
            model = XGBRegressor()
            model.load_model(weight_dir + 'model_{}.json'.format(i))
            model_lst.append(model)
        return model_lst

    def infer(self, item_id:int) -> Dict[str, Union[int, Dict[int, int]]]:
        preprocessed_data = self._preprocess(item_id)
        predictions = self._predict(preprocessed_data)
        if len(predictions) == 0:
            predicted_value = {}
            for shop_id in self.shops_df['shop_id'].unique().tolist():
                predicted_value[shop_id] = 0
            return {'item_id': item_id, 
                    'predicted_value': predicted_value}
        return {'item_id': item_id,
                'predicted_value': predictions.set_index('shop_id')['predicted_values'].to_dict()}
    
    def _data_preprocessing(self):
        last_train_data:pd.DataFrame = self.train_data[self.train_data['date_block_num'] > (33 - len(self.lag_frame))].copy()
        last_train_data = last_train_data[last_train_data['item_price'] > 0].reset_index(drop=True)
        last_train_data.loc[last_train_data['item_cnt_day'] < 0, 'item_cnt_day'] = 0
        last_train_data = last_train_data[(last_train_data['item_cnt_day'] < 1000)]
        last_train_data = last_train_data[last_train_data['item_price'] < 300000]


        last_train_data.sort_values(self.index_cols, inplace=True)
        item_cnt_month_group = last_train_data.groupby(self.index_cols)['item_cnt_day'].sum().reset_index()
        item_cnt_month_group.columns = self.index_cols + ['item_cnt_month']

        all_shop_train_data = []
        for i in range(34 - len(self.lag_frame), 34):
            sales = last_train_data[last_train_data['date_block_num']==i]
            all_shop_train_data.append(np.array(list(product([i],sales['shop_id'].unique(),sales['item_id'].unique())), dtype=np.int16))
        all_shop_train_data = pd.DataFrame(data=np.vstack(all_shop_train_data), columns=self.index_cols)
        all_shop_train_data.sort_values(self.index_cols, inplace=True)
        all_shop_train_data.reset_index(drop=True, inplace=True)
        all_shop_train_data = pd.merge(all_shop_train_data, item_cnt_month_group, on=self.index_cols, how='left')
        all_shop_train_data.fillna({'item_cnt_month':0},inplace=True)

        self.test_data['date_block_num'] = 34
        self.test_data.drop(columns='ID',inplace=True)
        full_data = pd.concat([all_shop_train_data, self.test_data]).copy()
        full_data = full_data.reset_index(drop=True)
        full_data.fillna({'item_cnt_month':0},inplace=True)

        full_data = pd.merge(full_data, self.items_df, on='item_id', how='left')

        lag_features = []
        drop_features = []

        group_item_id_by_month = full_data.groupby(['date_block_num', 'item_id'])['item_cnt_month'].aggregate(['mean'])
        group_item_id_by_month.columns = ['cnt_item_id_by_month']
        group_item_id_by_month.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_item_id_by_month, on=['date_block_num','item_id'], how='left')
        full_data['cnt_item_id_by_month'] = full_data['cnt_item_id_by_month'].astype('float16')
        lag_features += ['cnt_item_id_by_month']
        drop_features += ['cnt_item_id_by_month']

        group_shop_by_month = full_data.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].aggregate(['mean'])
        group_shop_by_month.columns = ['cnt_shop_id_by_month']
        group_shop_by_month.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_shop_by_month, on=['date_block_num','shop_id'], how='left')
        full_data['cnt_shop_id_by_month'] = full_data['cnt_shop_id_by_month'].astype('float16')
        lag_features += ['cnt_shop_id_by_month']
        drop_features += ['cnt_shop_id_by_month']

        group_item_category_by_month = full_data.groupby(['date_block_num', 'item_category_id'])['item_cnt_month'].aggregate(['mean'])
        group_item_category_by_month.columns = ['cnt_item_category_by_month']
        group_item_category_by_month.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_item_category_by_month, on=['date_block_num','item_category_id'], how='left')
        full_data['cnt_item_category_by_month'] = full_data['cnt_item_category_by_month'].astype('float16')
        lag_features += ['cnt_item_category_by_month']
        drop_features += ['cnt_item_category_by_month']

        group_item_category_by_shop_id_month = full_data.groupby(['date_block_num', 'shop_id', 'item_category_id'])['item_cnt_month'].aggregate(['mean'])
        group_item_category_by_shop_id_month.columns = ['cnt_item_category_by_shop_id_month']
        group_item_category_by_shop_id_month.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_item_category_by_shop_id_month, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
        full_data['cnt_item_category_by_shop_id_month'] = full_data['cnt_item_category_by_shop_id_month'].astype('float16')
        lag_features += ['cnt_item_category_by_shop_id_month']
        drop_features += ['cnt_item_category_by_shop_id_month']

        group_item_category_by_shop_id_month = full_data.groupby(['date_block_num', 'shop_id', 'item_id'])['item_cnt_month'].aggregate(['mean'])
        group_item_category_by_shop_id_month.columns = ['cnt_item_id_by_shop_id_month']
        group_item_category_by_shop_id_month.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_item_category_by_shop_id_month, on=['date_block_num', 'shop_id', 'item_id'], how='left')
        full_data['cnt_item_id_by_shop_id_month'] = full_data['cnt_item_id_by_shop_id_month'].astype('float16')
        lag_features += ['cnt_item_id_by_shop_id_month']
        drop_features += ['cnt_item_id_by_shop_id_month']

        group_price_mean = self.train_data.groupby(self.index_cols)['item_price'].aggregate(['mean'])
        group_price_mean.columns = ['item_price_mean']
        group_price_mean.reset_index(inplace=True)
        full_data = pd.merge(full_data, group_price_mean, on=self.index_cols, how='left')
        full_data['item_price_mean'] = full_data['item_price_mean'].astype('float16')
        drop_features += ['item_price_mean']

        group_price_mean.sort_values(by='date_block_num', inplace=True)
        first_price_per_item = group_price_mean.groupby(['shop_id', 'item_id'])['item_price_mean'].aggregate(['first'])
        first_price_per_item.columns = ['first_item_price_mean']
        first_price_per_item.reset_index(inplace=True)
        full_data = pd.merge(full_data, first_price_per_item, on=['shop_id', 'item_id'], how='left')
        full_data['first_item_price_mean'] = full_data['first_item_price_mean'].astype('float16')
        drop_features += ['first_item_price_mean']

        full_data['z_item_price_mean'] = full_data['item_price_mean'] / full_data['first_item_price_mean'] - 1.0
        full_data['z_item_price_mean'] = full_data['z_item_price_mean'].fillna(0)
        full_data['z_item_price_mean'] = full_data['z_item_price_mean'].astype('float16')
        lag_features += ['z_item_price_mean']
        drop_features += ['z_item_price_mean']

        full_data = self.create_lag_feature(full_data, lags=self.lag_frame, features=lag_features)
        full_data['month'] = full_data['date_block_num'] % 12 + 1
        full_data['shop_id'].astype("category")
        full_data['item_id'].astype("category")
        full_data['item_category_id'].astype("category")
        full_data['month'].astype("category")
        full_data['date_block_num'].astype("category")
        full_data.drop(drop_features, axis=1, inplace=True)
        full_data.fillna(0, inplace=True)
        features = ['date_block_num', 'month', 'shop_id', 'item_id', 'item_category_id']
        features += [f'{lag_feature}_lag_{i}' for lag_feature in lag_features for i in self.lag_frame]

        return full_data[full_data['date_block_num']==34][features]


    def _preprocess(self, item_id:int) -> pd.DataFrame:
        processed_data = self.processed_data[self.processed_data['item_id']==item_id].reset_index(drop=True)
        return processed_data
    
    def _predict(self, processed_data:pd.DataFrame) -> pd.DataFrame:
        result_lst = []
        for model in self.model_lst:
            test_preds = model.predict(processed_data)
            test_preds = np.clip(test_preds, 0, None).astype(np.int64)
            result_lst.append(test_preds)
        test_result = np.mean(np.stack(result_lst, axis=0), axis=0)
        processed_data['predicted_values'] = np.clip(test_result, 0, None).astype(np.int64)
        return processed_data

        
    def create_lag_feature(self, df, lags, features):
        return_df = df[df['date_block_num'] == 34].copy()
        for feature in features:
            # print(feature)
            for lag in lags:
                shifted = df[self.index_cols + [feature]].copy()
                shifted.columns = self.index_cols + [f'{feature}_lag_{lag}']
                shifted['date_block_num'] = shifted['date_block_num'] + lag
                return_df = pd.merge(return_df, shifted, on=self.index_cols, how='left')
        return return_df
    
if __name__ == '__main__':
    import time
    t = time.time()
    model = PredictionModel()
    print(time.time() - t)
    print(model.infer(0))
    print(model.infer(5822))