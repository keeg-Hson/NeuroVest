"""
Focal Loss Implementation for TensorFlow/Keras
===============================================
Focal loss is designed to address class imbalance by down-weighting easy examples
and focusing training on hard examples.

Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where:
- p_t is the model's estimated probability for the true class
- gamma (focus parameter): higher values increase focus on hard examples
  - gamma=0: equivalent to cross-entropy
  - gamma=2: typical value, good for most cases
- alpha: weight for positive class (to handle class imbalance)
  - alpha=0.27 (bullish proportion) for 73% bearish / 27% bullish split
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss for binary classification
    
    Parameters:
    -----------
    gamma : float, default=2.0
        Focusing parameter. Higher values give more weight to hard examples.
    alpha : float, default=0.25
        Weight for positive class. Should be set to proportion of positive class.
    from_logits : bool, default=False
        Whether y_pred is logits or probabilities
    """
    
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, name='focal_loss'):
        super().__init__(name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        """
        Compute focal loss
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted probabilities (or logits if from_logits=True)
        """
        # Convert logits to probabilities if needed
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        # Clip predictions to prevent log(0)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        # For positive class (y_true = 1)
        pos_loss = -self.alpha * K.pow(1.0 - y_pred, self.gamma) * K.log(y_pred)
        
        # For negative class (y_true = 0)
        neg_loss = -(1.0 - self.alpha) * K.pow(y_pred, self.gamma) * K.log(1.0 - y_pred)
        
        # Combine losses
        loss = y_true * pos_loss + (1.0 - y_true) * neg_loss
        
        return K.mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'from_logits': self.from_logits
        })
        return config


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """
    Functional API for focal loss
    
    Returns a loss function that can be passed to model.compile()
    
    Example:
        model.compile(
            optimizer='adam',
            loss=binary_focal_loss(gamma=2.0, alpha=0.27),
            metrics=['accuracy']
        )
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        pos_loss = -alpha * K.pow(1.0 - y_pred, gamma) * K.log(y_pred)
        neg_loss = -(1.0 - alpha) * K.pow(y_pred, gamma) * K.log(1.0 - y_pred)
        
        loss = y_true * pos_loss + (1.0 - y_true) * neg_loss
        return K.mean(loss)
    
    return focal_loss_fixed


# For model saving/loading compatibility
def get_focal_loss_config():
    """Returns config dict for custom objects in model.load_model()"""
    return {
        'FocalLoss': FocalLoss,
        'focal_loss_fixed': binary_focal_loss(gamma=2.0, alpha=0.27)
    }
