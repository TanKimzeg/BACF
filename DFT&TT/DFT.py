import numpy as np
from scipy.fft import fft, ifft
from sklearn.neural_network import MLPRegressor

class FourierTransformer:
    def __init__(self):
        self.mlp_filter = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    def dft(self, embeddings):
        """
        Perform Discrete Fourier Transform (DFT) on the embeddings.
        :param embeddings: 2D numpy array (n_samples, n_features)
        :return: Transformed embeddings in frequency domain
        """
        return fft(embeddings, axis=0)

    def idft(self, transformed_embeddings):
        """
        Perform Inverse Discrete Fourier Transform (IDFT) on the transformed embeddings.
        :param transformed_embeddings: 2D numpy array in frequency domain
        :return: Reconstructed embeddings in time domain
        """
        return ifft(transformed_embeddings, axis=0)

    def train_self_supervised_filter(self, embeddings):
        """
        Train the MLP-based filter in a self-supervised manner.
        :param embeddings: Original embeddings (2D numpy array)
        """
        # Perform DFT to get frequency domain data
        freq_data = self.dft(embeddings)

        # Use frequency domain data as input and original embeddings as target
        X_train = freq_data.real
        y_train = embeddings  # Original time-domain data as target

        # Train the filter
        self.mlp_filter.fit(X_train, y_train)

    def apply_filter(self, transformed_embeddings):
        """
        Apply the trained MLP-based filter to the frequency domain data.
        :param transformed_embeddings: 2D numpy array in frequency domain
        :return: Filtered frequency domain data
        """
        return self.mlp_filter.predict(transformed_embeddings)

    def denoise_and_decode(self, embeddings):
        """
        Denoise and decode the embeddings using the trained filter.
        :param embeddings: Original embeddings (2D numpy array)
        :return: Denoised and decoded embeddings
        """
        # Perform DFT to get frequency domain data
        freq_data = self.dft(embeddings)

        # Apply the trained filter
        filtered_freq_data = self.apply_filter(freq_data.real)

        # Perform IDFT to reconstruct time-domain data
        return self.idft(filtered_freq_data)

# Example usage
if __name__ == "__main__":
    # Generate example embeddings
    embeddings = np.random.rand(5, 4)  # 100 samples, 14 features
    print("Original Embeddings Shape:", embeddings.shape)
    print("Original Embeddings:", embeddings)

    transformer = FourierTransformer()

    # Train the filter in a self-supervised manner
    # transformer.train_self_supervised_filter(embeddings)

    # Denoise and decode the embeddings
    # denoised_embeddings = transformer.denoise_and_decode(embeddings)
    # print("Denoised Embeddings Shape:", denoised_embeddings.shape)
    # print("Denoised Embeddings:", denoised_embeddings)
    # print("Denoised Embeddings vs Original:", np.allclose(denoised_embeddings, embeddings, atol=1e-6))
    reconstructed_embeddings = transformer.idft(transformer.dft(embeddings))
    print("Reconstructed Embeddings Shape:", reconstructed_embeddings.shape)
    print("Reconstructed Embeddings:", reconstructed_embeddings)
    print("Reconstructed Embeddings vs Original:", np.allclose(reconstructed_embeddings, embeddings, atol=1e-6))