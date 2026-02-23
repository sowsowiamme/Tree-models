class Kmeans:
  def __init__(self, k, method, max_iter=1000, eps =1e-6):
    self.method = method
    self.k = k
    # the maximum iterations 
    self.max_iter = max_iter
    self.eps = eps

  def fit(self,X):
    # radomly select k center points，and it can only be implemented in one dimension
    indices = np.random.choice(X.shape[0], self.k, inplace=False)
    self.centers = X[indices]
    if self.method == "ecludiean":
      for i in range(self.iter):
        # all centers together using numpy broadcast
        dis = np.linalg.norm(X-center_matrix[:,None,:], axis =-1)
        labels = np.argmin(dis, axis = 1)
        new_centroids = np.zeros_like(self.centroids)
        for k in range(self.k):
          new_centroids[k] = X[labels == k].mean(axis=0)
        # 4. 检查收敛（中心变化小于 tol）
        shift = np.linalg.norm(new_centroids - self.centroids, axis=1).mean()
        self.centers = new_centroids
        if shift < self.tol:
            break
    return self
          
        
        
  
    
