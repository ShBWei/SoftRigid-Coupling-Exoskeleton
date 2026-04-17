"""
GangRouBingJi: Soft-Rigid Coupling Exoskeleton Prediction Network
Segmented spine architecture with dual-mode output (posture + rigidity)
"""
import torch
import torch.nn as nn

class SegmentAttention(nn.Module):
    """Cross-segment attention for Cervical-Thoracic-Lumbar-Sacral chain"""
    def __init__(self, hidden_dim, num_segments=4, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        # Spine connectivity mask: C↔T↔L↔S (adjacent segments strong coupling)
        self.register_buffer('spine_mask', self._create_spine_mask(num_segments))
        
    def _create_spine_mask(self, n):
        mask = torch.eye(n)
        for i in range(n-1):
            mask[i, i+1] = 0.8  # Adjacent coupling (e.g., Cervical-Thoracic)
            mask[i+1, i] = 0.8
        return mask.unsqueeze(0)  # [1, 4, 4]
    
    def forward(self, x):
        # x: [batch, 4_segments, hidden]
        masked_x = x * self.spine_mask
        attn_out, weights = self.attention(masked_x, masked_x, masked_x)
        return x + attn_out, weights  # Residual connection

class SoftRigidCouplingNet(nn.Module):
    """
    Dual-output network predicting:
    1. Posture angles (3-DOF per segment)
    2. Rigidity state (0=soft/flexible, 1=rigid/locked)
    """
    def __init__(self, 
                 num_segments=4,      # Cervical, Thoracic, Lumbar, Sacral
                 input_dim=6,         # IMU: acc(3) + gyro(3)
                 hidden_dim=64,
                 future_frames=20,    # 200ms @ 100Hz
                 num_heads=4):
        super().__init__()
        
        self.num_segments = num_segments
        self.future_frames = future_frames
        
        # Per-segment encoders (segmented architecture)
        self.segment_encoders = nn.ModuleList([
            nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
            for _ in range(num_segments)
        ])
        
        # Cross-segment attention (spine kinematic chain)
        self.cross_segment_attn = SegmentAttention(hidden_dim, num_segments, num_heads)
        
        # Temporal attention for motion prediction
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # Dual-head decoder
        # Head 1: Posture prediction (angles)
        self.posture_decoder = nn.Sequential(
            nn.Linear(hidden_dim * num_segments, 128),
            nn.ReLU(),
            nn.Linear(128, future_frames * num_segments * 3)  # 3 DOF: flex/ext, bend, rotate
        )
        
        # Head 2: Rigidity state (soft ↔ rigid switching)
        # Project Book: "毫秒级刚柔耦合动态切换"
        self.rigidity_decoder = nn.Sequential(
            nn.Linear(hidden_dim * num_segments, 64),
            nn.ReLU(),
            nn.Linear(64, future_frames * num_segments),  # 0=soft, 1=rigid
            nn.Sigmoid()  # Probability of rigid state
        )
        
    def forward(self, x, return_attention=False):
        """
        Input: [batch, time=40, segments=4, features=6]  (400ms @ 100Hz)
        Output: 
            posture: [batch, 20, 4, 3]  (angles in degrees)
            rigidity: [batch, 20, 4]    (0-1 probability)
        """
        batch_size = x.size(0)
        
        # Encode per segment
        segment_features = []
        for i, encoder in enumerate(self.segment_encoders):
            seg_input = x[:, :, i, :]  # [batch, time, 6]
            encoded, _ = encoder(seg_input)  # [batch, time, hidden]
            # Temporal attention for key frames
            attn_out, _ = self.temporal_attn(encoded, encoded, encoded)
            segment_features.append(attn_out[:, -1, :])  # Last timestep [batch, hidden]
        
        # Stack segments [batch, 4, hidden]
        seg_stack = torch.stack(segment_features, dim=1)
        
        # Cross-segment attention (spine coupling)
        coupled, cross_weights = self.cross_segment_attn(seg_stack)
        
        # Flatten for decoders
        flat = coupled.view(batch_size, -1)  # [batch, 4*hidden]
        
        # Decode
        posture_flat = self.posture_decoder(flat)
        rigidity_flat = self.rigidity_decoder(flat)
        
        # Reshape
        posture = posture_flat.view(batch_size, self.future_frames, 
                                   self.num_segments, 3)  # [B, 20, 4, 3]
        rigidity = rigidity_flat.view(batch_size, self.future_frames, 
                                     self.num_segments)  # [B, 20, 4]
        
        if return_attention:
            return posture, rigidity, cross_weights
        return posture, rigidity

if __name__ == "__main__":
    model = SoftRigidCouplingNet()
    dummy = torch.randn(2, 40, 4, 6)  # [batch, time, 4_segments, 6_imu]
    angles, rigid = model(dummy)
    print(f"Input: {dummy.shape}")
    print(f"Posture output: {angles.shape}")   # [2, 20, 4, 3]
    print(f"Rigidity output: {rigid.shape}")   # [2, 20, 4]
    print(f"Total params: {sum(p.numel() for p in model.parameters())/1e3:.1f}K")
