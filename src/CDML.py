import numpy as np


class CDML:
    
    def __init__(self, curves_count, input_dim, poly_degree, loss_function, regularization, reg_coef=0.0,
                 learning_rate=1e-3, n_iter=100, batch_size=100, silent=False, L=1_000, mu=1e-3):
        
        self.curves_count = curves_count # curves count, "m" in article
        self.input_dim = input_dim # input dimension, "d" in article
        self.poly_degree = poly_degree # polynomial degree, "c" in article
        self.loss_function = loss_function # loss function
        self.regularization = regularization # regularization function
        self.reg_coef = reg_coef # regularization coefficient
        self.learning_rate = learning_rate # learning rate
        self.n_iter = n_iter # iterations count
        self.batch_size = batch_size # batch size
        self.silent = silent # print loss
        self.L = L # integrals approximation accuracy
        self.mu = mu # # constant for smooth gradient approximation
    
        self.M = np.random.randn(self.curves_count, self.input_dim, self.poly_degree) # random tensor initialization
        self.history = [] # history dictionary, losses, tensors through iterations
        
        # precompute power values for Length(0, 1) calculation
        inter_values = np.arange(0, self.L) / self.L
        inter_values = np.repeat(inter_values.reshape(1, -1), self.poly_degree, axis=0)
        inter_values[0] = 1.0
        self.power_01_values = np.cumprod(inter_values, axis=0)
        
        self.curves_scalers = None # Length_i (0, 1) values, for fast transform after training
        
    # compute Length_i (T1, T2), i = curve_idx
    def __arc_length(self, T1, T2, curve_idx):
        
        T_min = min(T1, T2)
        T_max = max(T1, T2)
        dt = (T_max - T_min) / self.L
        inter_values = T_min + np.arange(0, self.L) * dt
        inter_values = np.repeat(inter_values.reshape(1, -1), self.poly_degree, axis=0)
        inter_values[0] = 1.0
        power_inter_values = np.cumprod(inter_values, axis=0)
        derivate_coefficients_with_const = self.M[curve_idx] * np.arange(1, self.poly_degree+1).reshape(1, -1)
        result = derivate_coefficients_with_const @ power_inter_values
        value = np.sqrt((result * result).sum(axis=0)).sum() * dt
        return value
    
    # compute projection T_i(x) on the i-th curve, i = curve_idx
    def __solve_ft(self, x, curve_idx):
        
        coefficients = np.zeros(shape=2*self.poly_degree)
        result_1 = self.M[curve_idx].T @ self.M[curve_idx]
        result_2 = -2 * self.M[curve_idx].T @ x.reshape(-1, 1)
        
        index_part = np.arange(1, self.poly_degree+1).reshape(1, -1) 
        index_sum = index_part + index_part.T
        for i in range(2, 2*self.poly_degree+1):
            coefficients[i-1] += result_1[index_sum == i].sum()
        coefficients[:self.poly_degree] += result_2.ravel()
        derivate_coefficients_with_const = coefficients * np.arange(1, 2*self.poly_degree+1)
        derivate = np.polynomial.Polynomial(derivate_coefficients_with_const)
        all_roots = derivate.roots()
            
        tor = 1e-7
        real_roots = all_roots[np.abs(all_roots.imag) < tor].real
        
        values = np.repeat(real_roots.reshape(1, -1), self.poly_degree, axis=0)
        power_values = np.cumprod(values, axis=0)
        
        Mts  = self.M[curve_idx] @ power_values
        results = Mts - x.reshape(-1, 1)
        real_roots_objs = (results * results).sum(axis=0).ravel()
        
        min_value = np.min(real_roots_objs)
        is_min = (real_roots_objs == min_value)
        minimizer_roots = real_roots[is_min]
        minimizer_root = np.min(minimizer_roots)
        return minimizer_root
    
    # gradient w.r.t. tensor M computed by one batch
    def __batch_gradient_squared_dist(self, X, X_hat):

        grads = np.zeros(shape=(self.batch_size, self.curves_count, self.input_dim, self.poly_degree))
        squared_dists = np.zeros(shape=self.batch_size)
        
        for curve_idx in range(self.curves_count):
            
            derivate_coefficients_with_const = self.M[curve_idx] * np.arange(1, self.poly_degree+1).reshape(1, -1)
            result = derivate_coefficients_with_const @ self.power_01_values
            length_norms = np.sqrt((result * result).sum(axis=0))
            length_unit = length_norms.sum() / self.L
            length_unit_grad = (result / length_norms.reshape(1, -1))  @ self.power_01_values.T / self.L
            
            for i in range(self.batch_size):
                
                x = X[i]
                x_hat = X_hat[i]
                
                delta_M = np.random.randn(self.input_dim, self.poly_degree)
                mu_delta_M = self.mu * delta_M 
                norm_delta = np.linalg.norm(delta_M) #, ord=2)
                
                cali_x = self.__solve_ft(x, curve_idx)
                cali_x_hat = self.__solve_ft(x_hat, curve_idx)
                length_cali = self.__arc_length(cali_x, cali_x_hat, curve_idx)
                length_cali = np.clip(length_cali, -5, 5)
                old_M = self.M[curve_idx].copy()
                self.M[curve_idx] += mu_delta_M
                cali_x_move = self.__solve_ft(x, curve_idx)
                cali_x_hat_move = self.__solve_ft(x_hat, curve_idx)
                length_cali_move = self.__arc_length(cali_x_move, cali_x_hat_move, curve_idx)
                self.M[curve_idx] = old_M.copy()
                
                cali_diff2 = (length_cali_move**2 - length_cali**2)
                cali_diff2 = np.clip(cali_diff2, -self.mu, self.mu)
                length_cali_grad2 = ( delta_M / (self.mu * (norm_delta**2)) ) * cali_diff2
                
                #abcd.append(length_cali)
                #print("!!", np.abs(self.M).max(), np.abs(length_cali_grad2).max(), np.abs(length_unit_grad).max(), cali_diff2, length_cali**2, length_unit**2)
                grads[i, curve_idx] = 2 * length_unit * length_unit_grad * (length_cali**2) + (length_unit**2) * length_cali_grad2
                squared_dists[i] += (length_unit * length_cali)**2
        return grads, squared_dists
    
    # generate batch
    def __generate_batch(self, X, y):
        elements_indices = np.random.permutation(X.shape[0])[:2*self.batch_size]
        batch = X[elements_indices].reshape(-1, 2, self.input_dim)
        labels = (y[elements_indices[::2]] == y[elements_indices[1::2]]).astype(float)
        return batch, labels
    
    # fit for one batch
    def __fit_batch(self, batch, labels):
        batch_grads_M, batch_squared_dists = self.__batch_gradient_squared_dist(batch[:, 0], batch[:, 1])
        #print("!!", batch_grads_M, self.M)
        batch_grads_squared_dist = self.loss_function.get_loss_grad(batch_squared_dists, labels)
        temp = batch_grads_squared_dist.reshape(self.batch_size, 1, 1, 1)
        grad_list = batch_grads_M * temp
        grad = grad_list.mean(axis=0)
        self.M = self.M - self.learning_rate * (grad + self.reg_coef * self.regularization.get_loss_grad(self))
    
    # fit function
    def fit(self, X, y):
        for iter_idx in range(self.n_iter):
            batch, labels = self.__generate_batch(X, y)
            self.__fit_batch(batch, labels)
        self.curves_scalers = np.empty(shape=self.curves_count)
        for curve_idx in range(self.curves_count):
            derivate_coefficients_with_const = self.M[curve_idx] * np.arange(1, self.poly_degree+1).reshape(1, -1)
            result = derivate_coefficients_with_const @ self.power_01_values
            length_norms = np.sqrt((result * result).sum(axis=0))
            length_unit = length_norms.sum() / self.L
            self.curves_scalers[curve_idx] = length_unit
    
    # get distance between x1 and x2 points with tensor M
    def get_dist(self, x1, x2):
        dist = 0
        for curve_idx in range(self.curves_count):
            dist += (self.curves_scalers[curve_idx]**2) * (self.__arc_length(x1[curve_idx], x2[curve_idx], curve_idx)**2)
        return np.sqrt(dist)
    
    # transform one point x
    def __transform_one(self, x):
        x_transformed = np.empty(shape=self.curves_count)
        for curve_idx in range(self.curves_count):
            x_transformed[curve_idx] = self.__solve_ft(x, curve_idx)
        return x_transformed
 
    # transform all points in X, result is vector of projections T_i(x)  
    def transform(self, X):
        X_transformed = []
        for x in X:
            X_transformed.append(self.__transform_one(x))
        return np.asarray(X_transformed, dtype=X.dtype)
    
    # fit + transform
    def fit_transform(self, X, y):
        self.fit(X, y)
        X_transformed = self.transform(X)
        return X_transformed