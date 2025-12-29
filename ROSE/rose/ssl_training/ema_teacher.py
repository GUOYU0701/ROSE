"""
Exponential Moving Average (EMA) Teacher for semi-supervised learning
"""
import torch
import torch.nn as nn
from typing import Optional
import copy


class EMATeacher(nn.Module):
    """
    Exponential Moving Average Teacher Model
    Maintains a slowly evolving copy of the student model for consistency training
    """
    
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999,
                 update_freq: int = 1, warmup_steps: int = 1000):
        super().__init__()
        
        self.ema_decay = ema_decay
        self.update_freq = update_freq
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        # Create teacher model as copy of student
        self.teacher_model = copy.deepcopy(student_model)
        
        # Freeze teacher model parameters
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        # Set teacher model to eval mode
        self.teacher_model.eval()
        
    def update_teacher(self, student_model: nn.Module, step: Optional[int] = None):
        """
        Update teacher model using EMA of student model parameters
        
        Args:
            student_model: Current student model
            step: Optional step count for decay scheduling
        """
        if step is not None:
            self.step_count = step
        else:
            self.step_count += 1
            
        # Skip update if not at update frequency
        if self.step_count % self.update_freq != 0:
            return
            
        # Calculate current decay rate
        current_decay = self._get_current_decay()
        
        # Update teacher parameters using EMA
        with torch.no_grad():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), 
                student_model.parameters()
            ):
                teacher_param.data = (
                    current_decay * teacher_param.data + 
                    (1.0 - current_decay) * student_param.data
                )
                
            # Update teacher buffers (batch norm running stats, etc.)
            for teacher_buffer, student_buffer in zip(
                self.teacher_model.buffers(),
                student_model.buffers()
            ):
                teacher_buffer.data = (
                    current_decay * teacher_buffer.data +
                    (1.0 - current_decay) * student_buffer.data
                )
    
    def _get_current_decay(self) -> float:
        """
        Get current EMA decay rate with warmup
        
        Returns:
            Current decay rate
        """
        if self.step_count < self.warmup_steps:
            # Linear warmup from 0 to ema_decay
            return self.ema_decay * (self.step_count / self.warmup_steps)
        else:
            return self.ema_decay
    
    def forward(self, *args, **kwargs):
        """Forward pass through teacher model"""
        return self.teacher_model(*args, **kwargs)
    
    def train(self, mode: bool = True):
        """Override train mode - teacher always in eval mode"""
        # Teacher model always stays in eval mode
        self.teacher_model.eval()
        return super().train(mode)
    
    def eval(self):
        """Set to eval mode"""
        self.teacher_model.eval() 
        return super().eval()
    
    def state_dict(self, *args, **kwargs):
        """Return teacher model state dict"""
        return self.teacher_model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        """Load teacher model state dict"""
        return self.teacher_model.load_state_dict(state_dict, *args, **kwargs)
    
    def get_teacher_model(self) -> nn.Module:
        """Get the teacher model"""
        return self.teacher_model
    
    def reset_teacher(self, student_model: nn.Module):
        """Reset teacher model to match current student model"""
        self.teacher_model.load_state_dict(student_model.state_dict())
        self.step_count = 0
        
    def get_ema_decay_schedule(self, total_steps: int) -> list:
        """
        Get EMA decay schedule for visualization/monitoring
        
        Args:
            total_steps: Total training steps
            
        Returns:
            List of decay rates over training
        """
        schedule = []
        for step in range(total_steps):
            if step < self.warmup_steps:
                decay = self.ema_decay * (step / self.warmup_steps)
            else:
                decay = self.ema_decay
            schedule.append(decay)
        return schedule


class AdaptiveEMATeacher(EMATeacher):
    """
    Adaptive EMA Teacher that adjusts decay rate based on performance
    """
    
    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999,
                 update_freq: int = 1, warmup_steps: int = 1000,
                 adaptation_rate: float = 0.1, performance_threshold: float = 0.1):
        super().__init__(student_model, ema_decay, update_freq, warmup_steps)
        
        self.base_ema_decay = ema_decay
        self.adaptation_rate = adaptation_rate
        self.performance_threshold = performance_threshold
        
        # Track performance for adaptation
        self.recent_performance = []
        self.performance_window = 10
        
    def update_teacher_with_performance(self, student_model: nn.Module, 
                                      performance_metric: float,
                                      step: Optional[int] = None):
        """
        Update teacher with performance-based decay adaptation
        
        Args:
            student_model: Current student model
            performance_metric: Current performance metric (e.g., validation loss)
            step: Optional step count
        """
        # Track recent performance
        self.recent_performance.append(performance_metric)
        if len(self.recent_performance) > self.performance_window:
            self.recent_performance.pop(0)
        
        # Adapt decay rate based on performance stability
        if len(self.recent_performance) >= 2:
            performance_change = abs(
                self.recent_performance[-1] - self.recent_performance[-2]
            )
            
            if performance_change < self.performance_threshold:
                # Performance is stable, increase decay (slower teacher updates)
                self.ema_decay = min(
                    0.999, 
                    self.ema_decay + self.adaptation_rate * 0.001
                )
            else:
                # Performance is changing, decrease decay (faster teacher updates)  
                self.ema_decay = max(
                    self.base_ema_decay * 0.9,
                    self.ema_decay - self.adaptation_rate * 0.001
                )
        
        # Update teacher model
        self.update_teacher(student_model, step)
        
    def get_current_decay_info(self) -> dict:
        """Get information about current decay adaptation"""
        return {
            'current_decay': self.ema_decay,
            'base_decay': self.base_ema_decay,
            'recent_performance': self.recent_performance,
            'performance_window': len(self.recent_performance)
        }