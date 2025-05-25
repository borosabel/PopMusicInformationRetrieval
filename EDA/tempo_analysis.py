import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

def calculate_tempo(file_path):
    """Calculate tempo for a given audio file using librosa's dynamic programming beat tracker."""
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Estimate tempo using dynamic programming
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        
        return tempo
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    # Load the dataframe
    df = pd.read_pickle('/Users/abelboros/Documents/Personal/Thesis/PopMusicInformationRetrieval/Data/pkl_data/preprocessed_df.pkl')
    
    # Create new file paths
    base_path = '/Users/abelboros/Documents/Personal/Thesis/PopMusicInformationRetrieval/Data/music'
    df['file_path'] = df['Path'].apply(lambda x: os.path.join(base_path, os.path.basename(x).replace('.mp3', '.wav')))
    
    # Calculate tempos
    print("Calculating tempos...")
    df['calculated_tempo'] = df['file_path'].apply(calculate_tempo)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Tempos by Coast
    sns.boxplot(data=df, x='Coast', y='Tempo1', ax=ax1)
    ax1.set_title('Original Tempo by Coast')
    ax1.set_xlabel('Coast')
    ax1.set_ylabel('Tempo (BPM)')
    
    sns.boxplot(data=df, x='Coast', y='calculated_tempo', ax=ax2)
    ax2.set_title('Calculated Tempo by Coast')
    ax2.set_xlabel('Coast')
    ax2.set_ylabel('Tempo (BPM)')
    
    plt.tight_layout()
    plt.savefig('Data/tempo_analysis.png')
    print("Analysis complete. Results saved to 'Data/tempo_analysis.png'")
    
    # Print statistics
    print("\nOriginal Tempo (Tempo1) Statistics by Coast:")
    print(df.groupby('Coast')['Tempo1'].describe())
    print("\nCalculated Tempo Statistics by Coast:")
    print(df.groupby('Coast')['calculated_tempo'].describe())

if __name__ == "__main__":
    main() 