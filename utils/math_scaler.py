import numpy as np
import json
import os
import torch

class MathScaler:
    """
    A specialized scaler for mathematical problems that handles the wide range of values
    better than standard normalization.
    """
    
    def __init__(self, method='log_scale'):
        """
        Initialize the math scaler.
        
        Args:
            method: 'log_scale', 'no_scale', or 'clip_scale'
        """
        self.method = method
        self.is_fitted = False
        
        # For log scaling
        self.log_offset = None
        self.sign_multiplier = None
        
        # For clipping
        self.min_val = None
        self.max_val = None
        self.scale_factor = None
    
    def fit(self, data):
        """
        Fit the scaler on training data.
        
        Args:
            data: numpy array or torch tensor of shape (n_samples, n_features)
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        if self.method == 'log_scale':
            self._fit_log_scale(data)
        elif self.method == 'clip_scale':
            self._fit_clip_scale(data)
        elif self.method == 'no_scale':
            self._fit_no_scale(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.is_fitted = True
    
    def _fit_log_scale(self, data):
        """Fit log scaling parameters."""
        # Handle negative values by using sign and absolute value
        abs_data = np.abs(data)
        
        # Add small offset to handle zeros
        abs_data = np.maximum(abs_data, 1e-8)
        
        # Log scale the absolute values
        log_data = np.log(abs_data)
        
        # Store parameters
        self.log_offset = np.mean(log_data)
        self.log_std = np.std(log_data)
    
    def _fit_clip_scale(self, data):
        """Fit clipping and scaling parameters."""
        # Use percentiles to avoid extreme outliers
        self.min_val = np.percentile(data, 1)  # 1st percentile
        self.max_val = np.percentile(data, 99)  # 99th percentile
        
        # Scale to [-1, 1] range
        self.scale_factor = 2.0 / (self.max_val - self.min_val)
    
    def _fit_no_scale(self, data):
        """No scaling - just store for consistency."""
        pass
    
    def transform(self, data):
        """
        Apply scaling to data.
        
        Args:
            data: numpy array or torch tensor
            
        Returns:
            Scaled data in the same format as input
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
        
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == 'log_scale':
            scaled = self._transform_log_scale(data)
        elif self.method == 'clip_scale':
            scaled = self._transform_clip_scale(data)
        elif self.method == 'no_scale':
            scaled = data
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if is_torch:
            return torch.tensor(scaled, dtype=torch.float32, device=device)
        return scaled
    
    def _transform_log_scale(self, data):
        """Apply log scaling transformation."""
        signs = np.sign(data)
        abs_data = np.abs(data)
        abs_data = np.maximum(abs_data, 1e-8)
        
        log_data = np.log(abs_data)
        # Standardize log values
        scaled_log = (log_data - self.log_offset) / self.log_std
        # Apply sign
        scaled = scaled_log * signs
        
        return scaled
    
    def _transform_clip_scale(self, data):
        """Apply clipping and scaling transformation."""
        # Clip to percentiles
        clipped = np.clip(data, self.min_val, self.max_val)
        
        # Scale to [-1, 1]
        scaled = (clipped - self.min_val) * self.scale_factor - 1.0
        
        return scaled
    
    def inverse_transform(self, data):
        """
        Apply inverse scaling to data.
        
        Args:
            data: numpy array or torch tensor
            
        Returns:
            Descaled data in the same format as input
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
        
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == 'log_scale':
            descaled = self._inverse_transform_log_scale(data)
        elif self.method == 'clip_scale':
            descaled = self._inverse_transform_clip_scale(data)
        elif self.method == 'no_scale':
            descaled = data
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if is_torch:
            return torch.tensor(descaled, dtype=torch.float32, device=device)
        return descaled
    
    def _inverse_transform_log_scale(self, data):
        """Apply inverse log scaling transformation."""
        signs = np.sign(data)
        abs_scaled = np.abs(data)
        
        # Reverse the standardization
        log_data = abs_scaled * self.log_std + self.log_offset
        abs_data = np.exp(log_data)
        
        # Restore sign
        descaled = abs_data * signs
        
        return descaled
    
    def _inverse_transform_clip_scale(self, data):
        """Apply inverse clipping and scaling transformation."""
        # Reverse scaling from [-1, 1] to [min_val, max_val]
        clipped = (data + 1.0) / self.scale_factor + self.min_val
        
        return clipped
    
    def save(self, filepath):
        """Save scaler parameters to a JSON file."""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before saving")
        
        params = {
            'method': self.method,
            'is_fitted': self.is_fitted,
            'log_offset': self.log_offset.tolist() if self.log_offset is not None else None,
            'log_std': self.log_std.tolist() if self.log_std is not None else None,
            'min_val': self.min_val.tolist() if self.min_val is not None else None,
            'max_val': self.max_val.tolist() if self.max_val is not None else None,
            'scale_factor': self.scale_factor.tolist() if self.scale_factor is not None else None
        }
        
        # Only create directory if the filepath has a directory component
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    def load(self, filepath):
        """Load scaler parameters from a JSON file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        
        self.method = params['method']
        self.is_fitted = params['is_fitted']
        self.log_offset = np.array(params['log_offset']) if params['log_offset'] is not None else None
        self.log_std = np.array(params['log_std']) if params['log_std'] is not None else None
        self.min_val = np.array(params['min_val']) if params['min_val'] is not None else None
        self.max_val = np.array(params['max_val']) if params['max_val'] is not None else None
        self.scale_factor = np.array(params['scale_factor']) if params['scale_factor'] is not None else None
