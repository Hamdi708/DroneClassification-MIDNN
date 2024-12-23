# DroneClassification-MIDNN



---

## Introduction
Drones (UAVs) have gained popularity due to their versatility in applications like surveillance, delivery, and smart agriculture. However, their misuse poses security risks such as espionage and attacks on critical infrastructure. Traditional detection methods like radar and audio are limited in identifying small UAVs. RF signals offer a promising alternative for accurate drone detection and classification.

---

## Literature Review: RF Characteristics for Classification
- RF fingerprints are unique features of RF signals emitted by electronic devices.
- Spectrogram images, derived from RF signals, have proven effective in drone classification when combined with convolutional neural networks (CNNs).

---

## Proposed Methodology
1. **Tools**:
   - PyTorch Lightning for simplified training.
   - Multi-Layer Perceptron (MLP) for learning hierarchical representations.
   - Convolutional Neural Network (CNN) for feature extraction from spectrogram images.
2. **Optimization**:
   - Best learning rate: `0.01` (88.5% accuracy).
   - Best optimizer: Adam (90% accuracy).
3. **Evaluation**:
   - Tested under varying SNR levels (-10 dB to 0 dB).
   - SinLU activation function performed best for this task.

---

## Contact
For more information, feel free to reach out.
