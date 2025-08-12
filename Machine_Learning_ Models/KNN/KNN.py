import numpy as np
from collections import Counter

class KNN:
    def __init__(self,X_train,y_train, k=1,  method="orsh"):
        self.k = k
        self.method = method
        self.X_train = X_train
        self.y_train = y_train


    def _calculate_distances(self, x):
        diffs = self.X_train - x
        square_num = np.square(diffs)
        # Rember axis, I always forget it......
        distance = np.sqrt(np.sum(square_num,axis=1))
        # My stucking point, how to link the label and the distance, 最后直接用一个元组存储
        return list(zip(distance, self.y_train))
    
    def _get_neighbors(self,distances):
        sorted_distances = sorted(distances, key=lambda x: x[0])
        return sorted_distances[:self.k]
    
    def _vote(self,sorted_distances):
        label_list = [label for _, label in sorted_distances]
        return Counter(label_list).most_common(1)[0][0]
    

    def fit(self, X, y):
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
    

    def predict(self,X_test):
        predictions = []
        for x in X_test:
            distances = self._calculate_distances(x)
            sorted_distances = self._get_neighbors(distances)
            predict_label = self._vote(sorted_distances)
            predictions.append(predict_label)
        return np.array(predictions)




    def _calculate_distances(self, x):
        """计算与所有训练样本的距离"""
        distances = []
        for i in range(len(self.X_train)):
            dist = np.sqrt(np.sum((x - self.X_train[i])**2))  # 欧氏距离
            distances.append((dist, self.y_train[i]))
        return distances
    

  




            

                
                

