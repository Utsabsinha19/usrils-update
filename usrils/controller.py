class RegulationController:
    """
    Implements the three-branch feedback Regulation Controller (Stage S5).
    Controls the learning rate (eta) based on the composite metric CIM (Theta_t).
    """
    def __init__(
        self,
        target_cim: float = 0.85,
        regulation_gain: float = 0.15,
        epsilon_ctrl: float = 0.02,
        min_lr: float = 0.001,
        max_lr: float = 0.500
    ):
        self.target_cim = target_cim
        self.regulation_gain = regulation_gain
        self.epsilon_ctrl = epsilon_ctrl
        self.min_lr = min_lr
        self.max_lr = max_lr

    def step(self, current_lr: float, current_cim: float):
        """
        Executes one step of the regulation logic.
        Returns (new_lr, branch_name).
        """
        delta_theta = current_cim - self.target_cim
        
        if delta_theta > self.epsilon_ctrl:
            # EXPAND: Accelerate (CIM exceeds target)
            factor = 1.0 + self.regulation_gain * abs(delta_theta)
            new_lr = min(current_lr * factor, self.max_lr)
            branch = 'EXPAND'
        elif abs(delta_theta) <= self.epsilon_ctrl:
            # HOLD: Stable (CIM near target)
            new_lr = current_lr
            branch = 'HOLD'
        else:
            # CONTRACT: Slow down (CIM below target)
            factor = 1.0 - self.regulation_gain * abs(delta_theta)
            new_lr = max(current_lr * factor, self.min_lr)
            branch = 'CONTRACT'
            
        return new_lr, branch
