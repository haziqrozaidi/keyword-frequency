import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

# Create a timestamped directory for results
def create_results_directory():
    """
    Create a directory structure to store results
    
    Returns:
        str: Path to the results directory
    """
    # Create main results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join('results', timestamp)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    return results_dir

# Set up logging with the timestamped directory
def setup_logging(results_dir):
    """
    Configure logging to write to a file in the results directory
    
    Args:
        results_dir (str): Path to the results directory
    """
    log_file = os.path.join(results_dir, 'keyword_analysis.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file
    )
    logging.info(f"Results will be saved to: {results_dir}")

def preprocess_text(text):
    """
    Preprocess the text by converting to lowercase, removing punctuation and stopwords
    
    Args:
        text (str): The input text to preprocess
        
    Returns:
        str: The preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def main():
    # Create results directory
    results_dir = create_results_directory()
    
    # Set up logging
    setup_logging(results_dir)
    
    logging.info("Starting keyword frequency analysis")
    
    # Download NLTK resources (only needed once)
    try:
        nltk.download('stopwords', quiet=True)
        logging.info("NLTK stopwords downloaded successfully")
    except Exception as e:
        logging.error(f"Error downloading NLTK stopwords: {e}")
    
    # Load the dataset
    try:
        df = pd.read_csv('../tweets_dataset.csv')
        logging.info(f"Dataset loaded successfully with {len(df)} rows")
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    # Define job sectors and associated keywords
    sector_keywords = {
        'finance': ['bank', 'investment', 'stock', 'crypto', 'economy', 'financial', 'money', 'trading', 'market', 'fund'],
        'healthcare': ['hospital', 'nurse', 'doctor', 'patient', 'vaccine', 'medical', 'health', 'clinic', 'care', 'treatment'],
        'education': ['school', 'teacher', 'student', 'university', 'curriculum', 'learning', 'education', 'teaching', 'academic', 'college'],
        'technology': ['ai', 'automation', 'robot', 'software', 'data', 'tech', 'digital', 'computer', 'algorithm', 'programming'],
        'manufacturing': ['factory', 'supply chain', 'production', 'assembly', 'labor', 'manufacturing', 'machine', 'industrial', 'plant', 'worker']
    }
    
    logging.info("Preprocessing tweet content")
    # Preprocess the tweet content
    df['processed_content'] = df['Content'].apply(preprocess_text)
    
    # Initialize columns for keyword frequency and binary presence
    for sector in sector_keywords:
        df[f'{sector}_keyword_count'] = 0
        df[f'{sector}_present'] = 0
    
    logging.info("Analyzing keyword frequency by sector")
    # Count keyword matches per sector for each tweet
    for index, row in df.iterrows():
        text = row['processed_content']
        for sector, keywords in sector_keywords.items():
            # Count keyword matches
            count = sum(1 for keyword in keywords if keyword in text.split())
            df.at[index, f'{sector}_keyword_count'] = count
            
            # Binary presence (1 if any keyword from sector appears)
            df.at[index, f'{sector}_present'] = 1 if count > 0 else 0
    
    # Create a summary table of total keyword matches per sector
    summary = {}
    for sector in sector_keywords:
        total_matches = df[f'{sector}_keyword_count'].sum()
        tweets_with_sector = df[df[f'{sector}_present'] == 1].shape[0]
        summary[sector] = {
            'total_keyword_matches': total_matches,
            'tweets_with_sector_keywords': tweets_with_sector,
            'percentage_of_tweets': round(tweets_with_sector / len(df) * 100, 2)
        }
    
    summary_df = pd.DataFrame(summary).T
    logging.info("Summary table created successfully")
    
    # Visualize the results
    try:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=summary_df.index, y=summary_df['total_keyword_matches'])
        plt.title('Total Keyword Matches per Sector')
        plt.ylabel('Number of Matches')
        plt.xlabel('Job Sector')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sector_keyword_matches.png'))
        logging.info("Visualization saved as sector_keyword_matches.png")
        
        # Create percentage visualization
        plt.figure(figsize=(12, 6))
        sns.barplot(x=summary_df.index, y=summary_df['percentage_of_tweets'])
        plt.title('Percentage of Tweets with Keywords from Each Sector')
        plt.ylabel('Percentage of Tweets (%)')
        plt.xlabel('Job Sector')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sector_tweet_percentage.png'))
        logging.info("Percentage visualization saved as sector_tweet_percentage.png")
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
    
    # Save the enriched DataFrame to a new CSV file
    try:
        df.to_csv(os.path.join(results_dir, 'tweets_with_sector_analysis.csv'), index=False)
        logging.info("Enriched dataset saved to tweets_with_sector_analysis.csv")
    except Exception as e:
        logging.error(f"Error saving enriched dataset: {e}")
    
    # Save the summary table to a CSV file
    try:
        summary_df.to_csv(os.path.join(results_dir, 'sector_keyword_summary.csv'))
        logging.info("Summary table saved to sector_keyword_summary.csv")
        
        # Create a README file with run information
        with open(os.path.join(results_dir, 'README.txt'), 'w') as f:
            f.write(f"Keyword Frequency Analysis\n")
            f.write(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Processed {len(df)} tweets across {len(sector_keywords)} job sectors\n\n")
            f.write("Summary of Results:\n")
            f.write(summary_df.to_string())
        
        print("\nKeyword frequency analysis completed successfully!")
        print(f"Processed {len(df)} tweets across {len(sector_keywords)} job sectors")
        print(f"Results saved to: {results_dir}")
    except Exception as e:
        logging.error(f"Error saving summary table: {e}")

if __name__ == "__main__":
    main()
