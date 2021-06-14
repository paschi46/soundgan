# SoundGAN
## Adversarial Synthesis of Audio with Dilated Deep Convolutional Generative Adversarial Network (Dilated DCGAN)

This python code is for importing, pre-processing and feeding AUDIO data (in .wav format) into a Dilated DCGAN model. Once trained, the model can generate high-quality samples of whatever audio material has been introduced (An example of this would be short drum or instrument stabs). This model has been optimized for convergence and rapid learning for short drum samples. Some results will be uploaded into the repository for anyone curious about the potential of this deep learning algorithm.

The model can potentially take audio of any length but the code needs to be changed and tuned for this as the model, as defined here, will only allow a length of 7500 samples.

The model can take raw, unprocessed audio as imput as preprocessing functions are all present.
