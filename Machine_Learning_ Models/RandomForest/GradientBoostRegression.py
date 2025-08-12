#gradientBoost  用决策树去拟合伪残差
import numpy as np

class TreeNode:

    def __init__(self, predict_vlaue = None, left= None, right=None, feature=None, threshold = None):
        self.value = predict_vlaue
        self.left = left
        self.right = right
        self.threshhold = threshold
        self.feature = feature


class GradientBoost:


    def __init__(self, depth, theta, max_n_trees, min_samples_split):
        self.max_depth = depth
        self.theta = theta
        self.max_n_trees = max_n_trees
        self,min_samples_split = min_samples_split
    
    def inital_prediction(y):
        y_inital = y.mean()
        return y_inital
    

    def _mse(self,r):
        return np.mean((r-np.mean(r))**2)

    def _find_best_split(self,X, r):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        for feature_idx in range(n_features):
            threshold_list = np.sort_values(X[:, feature_idx])
            threshold_list_value = (threshold_list[:-1], + threshold_list[1:]) /2
            for threshold in threshold_list_value:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                                # 加权平均MSE
                mse_left = self._mse(r[left_mask])
                mse_right = self._mse(r[right_mask]) 
                n_left, n_right = np.sum(left_mask), np.sum(right_mask)
                total_mse = (n_left * mse_left + n_right * mse_right) / (n_left + n_right)
                if total_mse <= best_mse:
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_mse = total_mse
        return best_feature, best_threshold, best_mse
    


    def _build_decision_tree(self, X, r, depth):
        if depth == self.max_depth :
            return TreeNode(value=np.mean(r))
        best_feature,best_threshold, best_mse = self._find_best_split(X, r)
        left_mask = X[:,best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._buiild_tree(X[left_mask], r[left_mask], depth+1)
        right_tree = self._buiild_tree(X[right_mask], r[right_mask], depth+1)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        
        # 初始化预测值 F0：所有样本的预测值相同（对于回归问题，通常是y的均值）
        self.initial_prediction = np.mean(y)
        F = np.array(self.initial_prediction(y)*len(y))
        
        for m in range(self.n_estimators):
            # 计算伪残差（负梯度）—— 一维数组
            # 用决策树拟合伪残差
            residuals = y- F
            tree = self._build_decision_tree(X, residuals, depth=self.max_depth)
            tree.fit(X, residuals)  # 注意：这里residuals是一维数组
            
            # 更新预测值：F = F + learning_rate * 新树的预测
            F += self.theta * tree.predict(X)  # tree.predict(X) 返回一维数组
            
            # 存储树
            self.trees.append(tree)
        return self
            




