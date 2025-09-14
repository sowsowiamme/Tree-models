import math
import numpy as np
from collections import Counter

class TreeNode:
    def __init__(self,value=None, left=None, right=None, threshold=None, feature=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature = feature 
        self.threshold = threshold
    def is_leaf(self):
        return self.left is None and self.right is None 
        
        
class RandomForestClassifier:

    def __init__(self, n_trees,max_depth, min_samples_split, method = "gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.method = method
    

    def _entropy(self, y):
        if len(y) ==0:
            return 0 
        p1 = y.sum() / len(y)
        p0 = 1 - p1
        return -(p1 *math.log(p1)+p0*math.log(p0))

    def _gini_impurity(self,y):
        if len(y) == 0:
            return 0
        p1 = y.sum() / len(y)
        p0 = 1 - p1
        return 1 - p0 **2 - p1 **2

    def find_best_split(self, X, y,feature_indices):
        # 在·随机森林的实现中会略有不同，需要存储原有的特征索引
        n_samples, n_features = X.shape
        best_feature = None
        best_impurity = float('inf')
        best_threshold = None
        for feature_idx in feature_indices:
            column_value = X[:,feature_idx].sort_values()
            unique_value_list = (column_value[:-1] + column_value[1:]) /2
            for threshold in unique_value_list:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                total_n = len(y)
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                if self.method == "gini":
                    left_impurity = self._gini_impurity(y[left_mask])
                    right_impurity = self._gini_impurity(y[right_mask])
                elif self.method == "entropy":
                    left_impurity = self._entropy(y[left_mask])
                    right_impurity = self._entropy(y[right_mask])
                weighted_average_impurity = left_mask.sum()/ total_n * left_impurity + right_mask.sum() /total_n * right_impurity
                if weighted_average_impurity < best_impurity:
                    best_impurity = weighted_average_impurity
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold, best_impurity



    def _build_tree(self,X,y, depth, feature_indices):

        if depth < self.depth or len(X) < 2* self.min_samples_split or len(np.unique(y)) == 1:
            majority_class = np.argmax(np.bincount(y))
            return TreeNode(value=majority_class)
        best_feature, best_threshold, best_impurity = self.find_best_split(X,y, feature_indices)
        if best_feature is None:
            majority_class = np.argmax(np.bincount(y))
            return TreeNode(value=majority_class)
        left_mask = X[:,best_feature] <= best_threshold
        right_mask = ~left_mask
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth+1, feature_indices)
        right_tree = self._build_tree(X[right_mask], y[right_mask],depth+1, feature_indices)
        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_tree, right=right_tree)
    

    def fit(self,X,y):
        # bootstrap 中对行和列都要做筛选
        self.tree_list = []
        for tree in range(self.n_trees):
            n_samples = X.shape[0]
            # bootstrap， 样本是有放回的
            row_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            n_features = X.shape[1]
            n_samples = max(1, int(n_features * self.feature_sample_rate))
            feature_idx = np.random.choice(n_features, size= n_samples, replace = False)
            # 特征是不放回的
            tree = self._build_tree(X[row_indices][:,feature_idx], y[row_indices],depth=0, feature_indices=feature_idx)
            self.tree_list.append(tree)
        return self
    

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        

        if x[node.feture] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    
    def predict(self, X):

        predictions = []
        for x in X:
            tree_preds = [self._traverse_tree(x, tree) for tree in self.trees]
            predictions.append(Counter(tree_preds).most_common(1)[0][0])
        return np.array(predictions)




# the second time Random Forest 
class TreeNode:

    def __init__(self,val=None, left=None, right=None, threshold=None, feature=None):
        self.val = val 
        self.left = left
        self.right = right
        self.threshold = threshold
        self.feature = feature
    def is_leaf(self):
        # 两边都是None 则返回None
        return self.left is None  and self.right is None


#  随机森林的过程实现， 重点是要注意bagging， feature的保存

class RandomForestClassifier:

    def __init__(self, n_trees=100, min_samples_split=3, max_depth = 3):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth


    def _entropy(self, y):
        "entropy p0* math.log(1/p0)"
        p1 = y.sum()/ len(y)
        p0 = 1 - p1
        entropy = p0*math.log(1/p0) + p1*math.log(1/p1)
        return entropy


    def _gini_impurity(self,y):
        ""
        p1 = y.sum() / len(y)
        p0 = 1 - p1
        gini_impurity = 1 - p0 ** 2- p1**2
        return gini_impurity

    def _find_best_split(self, X,y，feature_list, method="gini_impurity"):
        # 原有的feature位置要进行保留
        best_impurity = float('inf')
        best_threshold = None
        best_feature = None
        for feature_idx in feature_list:
            unique_threshold = np.unique(X[:, feature_idx]).sort_values()
            thresholds = (unique_threshold[:-1] + unique_threshold[1:]) / 2  #   取中点作为阈值
            for threshold in thresholds:
            # 获取该特征的所有唯一值作为候选阈值
                left_mask = X[:,feature_idx] <= threshold 
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
                    continue
                left_y = y[left_mask]
                right_y = y[right_mask]
                if self.method == "entropy":
                    avg_impurity = len(left_y)/ len(y) * self._entropy(left_y)  + len(right_y)/len(y) * self._entropy(right_y)
                elif method == "gini_impurity":
                    avg_impurity = len(left_y)/ len(y) * self._gini_impurity(left_y)  + len(right_y)/len(y) * self._gini_impurity(right_y)
                if avg_impurity < best_impurity:
                    best_feature = feature_idx
                    best_impurity = avg_impurity 
                    best_threshold = threshold
        return best_feature, best_impurity, best_threshold

    def _build_tree(self, X,y,feature_list, depth =0):
        if len(X) <= 2*self.min_samples_split or len(np.unique(y)) == 1:
            # 少数服从多数，多数的结果成为最终的预测值
            return TreeNode(val=Counter(y).most_common(1)[0][0])
        best_feature, best_impurity, best_threshold = self._find_best_split(X,y, feature_list)
        left_mask = X[: best_feature] <= best_threshold 
        right_mask = ~left_mask
        if np.sum(left_mask) < self.min_samples_split or np.sum(right_mask) < self.min_samples_split:
            continue 
        # 如果没有找到合适的分割，返回叶节点
        if best_feature is None:
            return TreeNode(val=Counter(y).most_common(1)[0][0])
        left_tree = self._build_tree(X[left_mask], y[left_mask], feature_list, depth +1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], feature_list, depth+1)
        return TreeNode(left_left_tree, right=right_tree, feature=best_feature, threshold =best_threhold)
        
        

    def fit(self, X):
        "模型构建过程"
        self.result_tree_list = []
        for tree in range(self.n_trees):
            n_samples, n_features = X.shape
            chosen_indices = np.random.choice(n_samples,n_samples, replace=True)
            X_sample = X[chosen_indices]
            y_sample = y[chosen_indices]
            chosen_features =  np.sqrt(n_features)
            feature_list = np.random.choice(X.features, chosen_features, inplace=False)
            tree = self._build_tree(X_sample,y_sample,feature_list)
            self.result_tree_list.append(tree)
        return self

     def _traverse_tree(self, x, tree):
        """遍历单棵树进行预测"""
        if tree.is_leaf():
            return tree.val
        
        if x[tree.feature] <= tree.threshold:
            return self._traverse_tree(x, tree.left)
        else:
            return self._traverse_tree(x, tree.right)
            
    def predict(self,X):
        "用每一棵树，对每一个构建的y值进行投票"
        predictions = []
        for row in range(len(X)):
            for tree in self.tree_list:
                y_this_sample = []
            
                y_this_sample.append(self._transverse_tree(X[row], tree))
            predictions = Counter(y_this_tree).most_common(1)[0][0]
            
        return np.array(predictions)
            
        
            
            
            
            
                
                
                
        


    
        
        
        
        








                


                


            

                
                
            

        
