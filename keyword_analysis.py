import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from datetime import datetime
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Set styling for plots
plt.style.use('ggplot')
sns.set_palette("tab10")

# Define the target keywords to track
TARGET_KEYWORDS = [
    # Technology & AI
    "ai", "artificial intelligence", "automation", "robot", "machine learning",
    "chatgpt", "deep learning", "algorithm", "tech", "digital transformation",
    
    # Jobs & Economy
    "job", "job loss", "layoff", "unemployment", "career",
    "workforce", "employment", "labour", "salary", "gig economy",
    
    # Skills & Reskilling
    "reskill", "upskill", "training", "education", "digital skills",
    "lifelong learning", "certification", "university",
    
    # Policy & Regulation
    "policy", "government", "regulation", "framework", "ethics",
    "responsibility", "governance",
    
    # Societal Impact & Emotion
    "future of work", "opportunity", "threat", "hope", "fear",
    "inequality", "trust", "surveillance", "bias", "privacy"
]

def load_data(file_path):
    """Load the CSV dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with {df.shape[0]} tweets and {df.shape[1]} columns.")
    return df

def preprocess_text(text):
    """Preprocess the tweet text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags (#tag)
    text = re.sub(r'#\w+', '', text)
    
    # Remove emojis (simplified approach)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def tokenize_and_remove_stopwords(text):
    """Tokenize text and remove stopwords."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        # Handle different possible date formats
        date_formats = [
            "%B %d, %Y at %I:%M %p",
            "%b %d, %Y at %I:%M %p",
            "%Y-%m-%d %H:%M:%S"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # If no format matched, try extracting just the date part
        match = re.search(r'([A-Za-z]+ \d+, \d+)', date_str)
        if match:
            date_part = match.group(1)
            return datetime.strptime(date_part, "%B %d, %Y")
            
        return pd.NaT
    except:
        return pd.NaT

def count_target_keywords(tokens):
    """Count occurrences of target keywords in tokens."""
    counts = {keyword: 0 for keyword in TARGET_KEYWORDS}
    
    # Convert tokens to a single string for easier multi-word keyword matching
    text = " ".join(tokens)
    
    for keyword in TARGET_KEYWORDS:
        # Count occurrences (case insensitive)
        counts[keyword] = len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        
    return counts

def analyze_keyword_frequency(df):
    """Analyze keyword frequency distribution."""
    all_tokens = []
    for tokens in df['tokens']:
        all_tokens.extend(tokens)
    
    # Get frequency distribution
    freq_dist = FreqDist(all_tokens)
    print("\nTop 20 most frequent words:")
    for word, count in freq_dist.most_common(20):
        print(f"{word}: {count}")
    
    return freq_dist

def analyze_with_tfidf(df):
    """Analyze keywords using TF-IDF."""
    # Create a TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=100)
    
    # Fit the vectorizer to the preprocessed tweets
    tfidf_matrix = tfidf.fit_transform(df['processed_text'])
    
    # Get feature names (words)
    feature_names = tfidf.get_feature_names_out()
    
    # Get TF-IDF scores
    tfidf_scores = tfidf_matrix.sum(axis=0).A1
    
    # Create a DataFrame with words and their TF-IDF scores
    tfidf_df = pd.DataFrame({'word': feature_names, 'tfidf': tfidf_scores})
    tfidf_df = tfidf_df.sort_values('tfidf', ascending=False)
    
    print("\nTop 20 keywords by TF-IDF score:")
    print(tfidf_df.head(20))
    
    return tfidf_df

def track_keyword_trends(df):
    """Track keyword trends over time."""
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        print("Error: No valid datetime column found.")
        return None
    
    # Create a new DataFrame with date and keyword counts
    trends_data = []
    
    for _, row in df.iterrows():
        if pd.isna(row['datetime']):
            continue
            
        date = row['datetime']
        counts = count_target_keywords(row['tokens'])
        
        for keyword, count in counts.items():
            if count > 0:
                trends_data.append({
                    'date': date,
                    'keyword': keyword,
                    'count': count
                })
    
    trends_df = pd.DataFrame(trends_data)
    
    # If no data found, return
    if trends_df.empty:
        print("No keyword occurrences found in the dataset.")
        return None
    
    return trends_df

def group_by_time_period(trends_df, period='month'):
    """Group keyword trends by month or quarter."""
    if trends_df is None or trends_df.empty:
        return None
        
    # Add month and quarter columns
    trends_df['month'] = trends_df['date'].dt.to_period('M')
    trends_df['quarter'] = trends_df['date'].dt.to_period('Q')
    
    # Group by the specified period
    group_col = 'month' if period == 'month' else 'quarter'
    grouped = trends_df.groupby([group_col, 'keyword'])['count'].sum().reset_index()
    
    # Convert period to string for plotting
    grouped[group_col] = grouped[group_col].astype(str)
    
    return grouped

def plot_keyword_trends(grouped_df, period='month'):
    """Plot keyword trends over time."""
    if grouped_df is None or grouped_df.empty:
        print(f"No data available for plotting {period}ly trends.")
        return
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Get unique keywords
    keywords = grouped_df['keyword'].unique()
    
    # Plot a line for each keyword
    for keyword in keywords:
        keyword_data = grouped_df[grouped_df['keyword'] == keyword]
        x_col = 'month' if period == 'month' else 'quarter'
        plt.plot(keyword_data[x_col], keyword_data['count'], marker='o', linewidth=2, label=keyword)
    
    # Add labels and title
    plt.xlabel(f'Time ({period})', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Keyword Frequency Trends Over Time ({period.capitalize()}ly)', fontsize=14)
    plt.legend(fontsize=10)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'keyword_trends_by_{period}.png', dpi=300)
    print(f"Plot saved as keyword_trends_by_{period}.png")
    
    plt.show()

def plot_keyword_heatmap(trends_df):
    """Create a heatmap of keyword frequency over time."""
    if trends_df is None or trends_df.empty:
        print("No data available for heatmap visualization.")
        return
    
    # Create pivot table: rows=time periods, columns=keywords, values=counts
    pivot_monthly = trends_df.pivot_table(
        index='month', 
        columns='keyword', 
        values='count', 
        aggfunc='sum',
        fill_value=0
    )
    
    # Set up the plot
    plt.figure(figsize=(16, 10))
    
    # Create the heatmap
    sns.heatmap(
        pivot_monthly, 
        annot=True, 
        fmt='g', 
        cmap='YlGnBu', 
        linewidths=0.5, 
        cbar_kws={'label': 'Frequency'}
    )
    
    # Add labels and title
    plt.title('Keyword Frequency Heatmap by Month', fontsize=16)
    plt.xlabel('Keywords', fontsize=12)
    plt.ylabel('Month', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('keyword_frequency_heatmap.png', dpi=300)
    print("Plot saved as keyword_frequency_heatmap.png")
    
    plt.show()

def plot_keyword_bar_chart(df, trends_df=None):
    """Create a bar chart of overall keyword frequencies."""
    plt.figure(figsize=(14, 8))
    
    if trends_df is not None and not trends_df.empty:
        # Aggregate keyword counts from trends data
        keyword_counts = trends_df.groupby('keyword')['count'].sum().reset_index()
        keyword_counts = keyword_counts.sort_values('count', ascending=False)
        
        # Plot bar chart
        sns.barplot(x='keyword', y='count', data=keyword_counts, palette='viridis')
        
    else:
        # Calculate keyword counts directly from tokens
        keyword_counts = {keyword: 0 for keyword in TARGET_KEYWORDS}
        
        for tokens in df['tokens']:
            text = " ".join(tokens)
            for keyword in TARGET_KEYWORDS:
                keyword_counts[keyword] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text))
        
        # Convert to DataFrame for plotting
        keyword_df = pd.DataFrame({
            'keyword': list(keyword_counts.keys()),
            'count': list(keyword_counts.values())
        })
        keyword_df = keyword_df.sort_values('count', ascending=False)
        
        # Plot bar chart
        sns.barplot(x='keyword', y='count', data=keyword_df, palette='viridis')
    
    # Add labels and title
    plt.title('Overall Frequency of Target Keywords', fontsize=16)
    plt.xlabel('Keywords', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('keyword_frequency_bar_chart.png', dpi=300)
    print("Plot saved as keyword_frequency_bar_chart.png")
    
    plt.show()

def plot_tfidf_bar_chart(tfidf_df, top_n=15):
    """Create a bar chart of top TF-IDF scores."""
    # Get top N words by TF-IDF score
    top_tfidf = tfidf_df.head(top_n)
    
    # Plot
    plt.figure(figsize=(14, 8))
    sns.barplot(x='tfidf', y='word', data=top_tfidf, palette='rocket', orient='h')
    
    plt.title(f'Top {top_n} Keywords by TF-IDF Score', fontsize=16)
    plt.xlabel('TF-IDF Score', fontsize=12)
    plt.ylabel('Words', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('tfidf_bar_chart.png', dpi=300)
    print("Plot saved as tfidf_bar_chart.png")
    
    plt.show()

def main():
    # Load data
    df = load_data('tweets_dataset.csv')
    
    # Inspect the data
    print("\nDataset sample:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    
    # Preprocess tweets
    print("\nPreprocessing tweets...")
    df['processed_text'] = df['Content'].apply(preprocess_text)
    df['tokens'] = df['processed_text'].apply(tokenize_and_remove_stopwords)
    
    # Parse dates
    print("Parsing dates...")
    df['datetime'] = df['Date'].apply(parse_date)
    
    # Analyze keyword frequency
    print("Analyzing keyword frequencies...")
    freq_dist = analyze_keyword_frequency(df)
    
    # TF-IDF Analysis
    print("Performing TF-IDF analysis...")
    tfidf_results = analyze_with_tfidf(df)
    
    # Track keyword trends
    print("Tracking keyword trends...")
    trends_df = track_keyword_trends(df)
    
    # Group by month and quarter
    print("Grouping data by month...")
    monthly_trends = group_by_time_period(trends_df, 'month')
    print("Grouping data by quarter...")
    quarterly_trends = group_by_time_period(trends_df, 'quarter')
    
    # Plot trends
    print("Plotting trends...")
    plot_keyword_trends(monthly_trends, 'month')
    plot_keyword_trends(quarterly_trends, 'quarter')
    
    # Create new visualizations
    print("Creating heatmap visualization...")
    plot_keyword_heatmap(monthly_trends)
    
    print("Creating bar chart visualizations...")
    plot_keyword_bar_chart(df, trends_df)
    plot_tfidf_bar_chart(tfidf_results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
