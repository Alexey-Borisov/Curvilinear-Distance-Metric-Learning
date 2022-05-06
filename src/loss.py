import abc
import numpy as np


class LossFunction(abc.ABC):
        
    @abc.abstractclassmethod
    def get_loss(self, squared_distances, labels):
        pass
    
    @abc.abstractclassmethod
    def get_loss_grad(self, squared_distances, labels):
        pass
    
    
class Regularization(abc.ABC):
    
    @abc.abstractclassmethod
    def get_loss(self, model):
        pass
    
    @abc.abstractclassmethod
    def get_loss_grad(self, model):
        pass
    
    
class L2Regularization(Regularization):
    
    def get_loss(self, model):
        return model.M**2
    
    def get_loss_grad(self, model):
        return 2 * model.M
    

class ContrastiveLoss(LossFunction):
    
    def __init__(self, similar_bound=1.0, dissimilar_bound=2.0):
        self.similar_bound = similar_bound
        self.dissimilar_bound = dissimilar_bound
    
    def get_loss(self, squared_distances, labels):
        
        similar_loss = labels * np.maximum(0, squared_distances - self.similar_bound)**2
        dissimilar_loss = (1 - labels) * np.maximum(0, self.dissimilar_bound - squared_distances)**2
        loss = similar_loss + dissimilar_loss
        return loss
    
    def get_loss_grad(self, squared_distances, labels):
        
        similar_loss_grad = 2 * labels * (squared_distances > self.similar_bound) * (squared_distances - self.similar_bound)
        dissimilar_loss_grad = -2 * (1 - labels) * (squared_distances < self.dissimilar_bound) * (self.dissimilar_bound - squared_distances)
        loss_grad = similar_loss_grad + dissimilar_loss_grad
        return loss_grad
    
                 