Traditional wireless channel estimation techniques such as Least Squares (LS) and Minimum 
Mean Square Error (MMSE) often fail in challenging real-world conditions — especially when 
pilots are sparse or channel characteristics change rapidly. These methods do not generalize 
well to noisy or dynamic environments and require accurate prior statistics to perform reliably.   
. Objectives:   
● To design a deep learning-based solution that can robustly estimate wireless   
MIMO-OFDM channels 
● To outperform traditional LS estimators under noisy, pilot-sparse conditions   
● To introduce modern architectural features (residual and attention blocks) for improved 
learning   
Dataset Used:   
We synthetically generated a dataset using simulated Rayleigh and Rician fading models.   
Key parameters:   
● MIMO Setup: 2x2 (2 TX, 2 RX antennas)   
● OFDM symbols: 14   
● Subcarriers: 64   
● Pilot spacing: every 4th subcarrier   
● SNR: 0–30 dB range   
● Modulation: QPSK   
Proposed Solution:   
We built a custom deep learning model: ResAtt-CNN   
It integrates:   
● Residual blocks to enable deeper learning and better gradient flow   
● Squeeze-and-Excitation (SE) attention to enhance important feature channels   
● Fully convolutional architecture trained to minimize MSE between estimated and true
channel response   
Training:   
● Optimizer: Adam   
● Loss: MSE   
● Epochs: 10   
● Batch size: 8   
5. Tools & Technology Used:   
● Language: Python   
● Framework: PyTorch   
● Execution: Google Colab with GPU  
 Visualizations: matplotlib   
● Model Saving: torch.save()   
6. Evaluation Results:   
Model   
NMSE @ 20dB   
LS Estimator ~1.00+   
Notes   
Poor at low SNR   
ResAtt-CNN 0.8489   
Visuals:   
Much better generalization   
● Training loss curve (included in screenshots)   
●  Heatmap of predicted vs true CSI (channel)   
Key Insights:   
● Deep learning-based estimation adapts better to noise and pilot sparsity   
● Attention helps model focus on more informative subcarrier patterns   
● Residual connections improve convergence without vanishing gradients  
ResAtt-CNN shows strong potential for integration in 5G/6G receivers 
Conclusion:   
This project demonstrates how a custom CNN with attention can outperform classical wireless   
signal processing techniques under harsh channel conditions. With further training and   
hyperparameter tuning, ResAtt-CNN can become a deployable lightweight module for modern    
wireless systems, including base stations, UAVs, and edge IoT devices.  
