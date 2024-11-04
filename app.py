import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import io

# Streamlit app title
st.title("Audio Spectrogram Generator")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# Sidebar options for spectrogram parameters
st.sidebar.header("Spectrogram Parameters")
n_fft = st.sidebar.slider("FFT window size (n_fft)", 256, 4096, 2048, step=256)
hop_length = st.sidebar.slider("Hop length", 64, 1024, 512, step=64)
cmap = st.sidebar.selectbox("Color map", ['viridis', 'plasma', 'inferno', 'magma', 'cividis'])

if uploaded_file is not None:
    # Load audio file
    audio, sr = librosa.load(uploaded_file, sr=None)
    
    # Generate spectrogram
    st.write("Generating spectrogram...")
    D = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Display spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log', cmap=cmap)
    #plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    
    # Save the spectrogram to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # Display the spectrogram on Streamlit
    st.pyplot(plt.gcf())
    
    # Button to download the spectrogram
    st.download_button(
        label="Download Spectrogram",
        data=buf,
        file_name="spectrogram.png",
        mime="image/png"
    )

    # Close the plot to prevent Streamlit from displaying it twice
    plt.close()
else:
    st.write("Please upload an audio file to proceed.")
