from utils import * 
from numpy import zeros
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


class SVM():
    def __init__(self, feature_list, kernel = 'rbf'):
        self.feature_list = feature_list
        self.kernel = kernel

    def predict(self, df_cross):
        df_cross.loc[:,self.feature_list] = self.scaler.transform(df_cross[self.feature_list])
        dfg = df_cross.groupby('file_name')
        pred_file_pid = pd.DataFrame(columns=('file_name', 'genre'))
        for song_name, group in dfg:
            x_test = group[self.feature_list]
            model = self.model
            output = model.predict(x_test)
            pred_file_pid = pred_file_pid.append(pd.DataFrame(data={
                'file_name':song_name,
                'genre':[np.argmax(np.bincount(output))]
            }))
        return pred_file_pid
    
    def fit(self, df_train):
        self.scaler = preprocessing.StandardScaler().fit(df_train[self.feature_list])
        df_train.loc[:,self.feature_list] = self.scaler.transform(df_train[self.feature_list])
    
        dfg = df_train.groupby('genre')
        x_train = df_train.loc[:,self.feature_list]
        y_train = list()
        for name, group in dfg:
            for i in range(len(group[self.feature_list])):
                 y_train.append(name)
        
        clf = OneVsRestClassifier(SVC(kernel=self.kernel, C=1., random_state=42))
        # clf = svm.SVC(decision_function_shape='ovo')
        model = clf.fit(x_train, y_train)
        self.model = model

    def predict_df(self, df_cross):
        pred_file_pid = self.predict(df_cross)
        
        pred_file_pid.sort_values('file_name', inplace=True)
        orig_file_pid = df_cross[['genre','file_name']].drop_duplicates().sort_values('file_name')
        # return metrics.classification_report(orig_file_pid['genre'], pred_file_pid['genre'])
        return orig_file_pid, pred_file_pid
        
    def predict_and_test(self, df_cross):
        pred_file_pid = self.predict(df_cross)

        pred_file_pid.sort_values('file_name', inplace=True)

        orig_file_pid = df_cross[['genre','file_name']].drop_duplicates().sort_values('file_name')
        print classification_report(orig_file_pid['genre'], 
                                        pred_file_pid['genre'])