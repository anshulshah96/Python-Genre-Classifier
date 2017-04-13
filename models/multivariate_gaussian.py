from utils import * 

from scipy.stats import multivariate_normal
from sklearn.metrics import classification_report
from numpy import zeros

class MultiGauss():
    def __init__(self, feature_list):
        self.feature_list = feature_list

    def predict(self, df_cross):
        df_cross.loc[:,self.feature_list] = self.scaler.transform(df_cross[self.feature_list])

        cls = len(self.mm)
        dfg = df_cross.groupby('file_name')
        pred_file_pid = pd.DataFrame(columns=('file_name', 'genre'))

        for file_name, group in dfg:
            total = group['genre'].count()
            dist = zeros((total,cls))
            for i in range(cls):
                dist[:,i] = multivariate_normal.pdf(group[self.feature_list], 
                            mean=self.mm[i], cov=self.covm[i],allow_singular=True)
            plist = np.sum(np.log(dist), axis =0)
            pred_file_pid = pred_file_pid.append(pd.DataFrame(data={
                'file_name':file_name,
                'genre':[int(self.ids[np.argmax(plist)])]
            }))
        return pred_file_pid
    
    def fit(self, df_train):
        self.scaler = preprocessing.StandardScaler().fit(df_train[self.feature_list])
        df_train.loc[:,self.feature_list] = self.scaler.transform(df_train[self.feature_list])
    
        dfg = df_train.groupby('genre')
        mm = list()
        covm = list()
        ids = list()
        for name, group in dfg:
            mm.append(np.mean(group[self.feature_list],axis=0))
            covm.append(np.cov(group[self.feature_list], rowvar=False))
            ids.append(int(name))

        self.mm = mm
        self.covm = covm
        self.ids = ids

    def predict_df(self, df_cross):
        pred_file_pid = self.predict(df_cross)

        pred_file_pid.sort_values('file_name', inplace=True)

        orig_file_pid = df_cross[['genre','file_name']].drop_duplicates().sort_values('file_name')
        return orig_file_pid, pred_file_pid

    def predict_and_test(self, df_cross):
        pred_file_pid = self.predict(df_cross)

        pred_file_pid.sort_values('file_name', inplace=True)

        orig_file_pid = df_cross[['genre','file_name']].drop_duplicates().sort_values('file_name')
        print classification_report(orig_file_pid['genre'], 
                                        pred_file_pid['genre'])