import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys
import logging
from datetime import datetime

def create_visualizations():
    """
    Create additional visualizations based on the enriched dataset
    """
    # Find the most recent results directory
    try:
        results_base_dir = 'results'
        if not os.path.exists(results_base_dir):
            print("Results directory not found. Please run keyword_analysis.py first.")
            sys.exit(1)
            
        # Get the most recent directory based on name (which is a timestamp)
        subdirs = [d for d in os.listdir(results_base_dir) if os.path.isdir(os.path.join(results_base_dir, d))]
        if not subdirs:
            print("No result directories found. Please run keyword_analysis.py first.")
            sys.exit(1)
            
        latest_dir = max(subdirs)
        results_dir = os.path.join(results_base_dir, latest_dir)
        
        # Set up logging
        log_file = os.path.join(results_dir, 'visualizations.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        logging.info(f"Creating additional visualizations in: {results_dir}")
        
        # Load the enriched dataset
        df = pd.read_csv(os.path.join(results_dir, 'tweets_with_sector_analysis.csv'))
        logging.info(f"Loaded enriched dataset with {len(df)} rows")
        
        # 1. Create a correlation heatmap between sectors
        sector_cols = [col for col in df.columns if col.endswith('_present')]
        sector_names = [col.split('_present')[0] for col in sector_cols]
        
        corr_df = df[sector_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                   xticklabels=sector_names, yticklabels=sector_names)
        plt.title('Correlation Between Job Sectors in Tweets')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sector_correlation.png'))
        logging.info("Created sector correlation heatmap")
        
        # 2. Create a stacked bar chart for sector distribution by engagement
        # Calculate average likes and retweets per sector
        engagement_data = []
        for sector in sector_names:
            sector_tweets = df[df[f'{sector}_present'] == 1]
            if not sector_tweets.empty:
                avg_likes = sector_tweets['Likes'].mean()
                avg_retweets = sector_tweets['Retweets'].mean()
                engagement_data.append({
                    'Sector': sector,
                    'Avg Likes': avg_likes,
                    'Avg Retweets': avg_retweets
                })
        
        engagement_df = pd.DataFrame(engagement_data)
        
        plt.figure(figsize=(12, 6))
        engagement_df.set_index('Sector').plot(kind='bar', stacked=False)
        plt.title('Average Engagement by Sector')
        plt.ylabel('Average Count')
        plt.xlabel('Job Sector')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'sector_engagement.png'))
        logging.info("Created sector engagement bar chart")
        
        # 3. Visualize top tweets by sector (based on total engagement)
        df['total_engagement'] = df['Likes'] + df['Retweets'] + df['Replies'] + df['Quotes']
        
        plt.figure(figsize=(14, 10))
        
        for i, sector in enumerate(sector_names):
            plt.subplot(3, 2, i+1)
            sector_tweets = df[df[f'{sector}_present'] == 1].copy()
            if not sector_tweets.empty:
                # Get top 10 tweets by engagement
                top_tweets = sector_tweets.nlargest(min(10, len(sector_tweets)), 'total_engagement')
                
                # Create a bar chart
                sns.barplot(x='total_engagement', y=top_tweets.index, data=top_tweets)
                plt.title(f'Top Tweets in {sector.capitalize()} Sector')
                plt.xlabel('Total Engagement')
                plt.ylabel('Tweet Index')
            
            if i >= 4:  # Only create 5 subplots (one for each sector)
                break
                
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'top_tweets_by_sector.png'))
        logging.info("Created top tweets visualization")
        
        # Create a variable to track if wordcloud directory was created
        wordcloud_dir = None
        
        # 4. Create a word cloud for each sector (using the most frequent keywords)
        try:
            from wordcloud import WordCloud
            
            # Create a subdirectory for word clouds
            wordcloud_dir = os.path.join(results_dir, 'wordclouds')
            if not os.path.exists(wordcloud_dir):
                os.makedirs(wordcloud_dir)
            
            for sector in sector_names:
                # Get tweets in this sector
                sector_tweets = df[df[f'{sector}_present'] == 1]['processed_content'].str.cat(sep=' ')
                
                # Generate word cloud
                if sector_tweets:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                                         max_words=100, contour_width=3).generate(sector_tweets)
                    
                    # Save the word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Word Cloud for {sector.capitalize()} Sector')
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(wordcloud_dir, f'{sector}_wordcloud.png'))
            
            logging.info("Created word clouds for each sector")
        except ImportError:
            logging.warning("WordCloud package not installed. Skipping word cloud generation.")
        
        # Update the README with visualization information
        with open(os.path.join(results_dir, 'README.txt'), 'a') as f:
            f.write("\n\nAdditional Visualizations Created:\n")
            f.write("- sector_correlation.png: Correlation heatmap between job sectors\n")
            f.write("- sector_engagement.png: Average engagement metrics by sector\n")
            f.write("- top_tweets_by_sector.png: Top tweets by engagement for each sector\n")
            if wordcloud_dir and os.path.exists(wordcloud_dir):
                f.write("- wordclouds/: Directory containing word clouds for each sector\n")
        
        print(f"\nAdditional visualizations created successfully in: {results_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        if 'logging' in sys.modules:
            logging.error(f"Error creating visualizations: {e}")

if __name__ == "__main__":
    create_visualizations()
