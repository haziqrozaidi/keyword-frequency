# Keyword Frequency Analysis

This project analyzes keyword frequency and trends in a Twitter dataset ('tweets_dataset.csv').

## Dataset Columns

- Tweet ID
- URL
- Content (tweet text)
- Likes
- Retweets
- Replies
- Quotes
- Views
- Date (timestamp)

## Target Keywords

The script specifically tracks the frequency of these keywords:
- automation
- job loss
- reskill
- AI
- robot
- future of work

## Features

1. **Data Loading and Preprocessing**:
   - Load CSV data
   - Convert text to lowercase
   - Remove URLs, mentions (@user), hashtags (#tag), emojis, and punctuation
   - Tokenize text
   - Remove English stopwords

2. **Keyword Analysis**:
   - Frequency distribution of all words
   - TF-IDF analysis to identify important terms
   - Specific tracking of target keywords

3. **Time-based Analysis**:
   - Group data by month and quarter
   - Track keyword frequency trends over time

4. **Visualization**:
   - Create line plots showing keyword frequency trends
   - One line per keyword to compare trends

## Setup and Usage

### Requirements

```
pip install -r requirements.txt
```

### Running the Analysis

```
python keyword_analysis.py
```

### Output

The script generates:
1. Console output with analysis results
2. Two visualization files:
   - `keyword_trends_by_month.png`
   - `keyword_trends_by_quarter.png`

## Results Interpretation

The line plots show how frequently each target keyword appears over time, helping to identify:

- Which topics are trending upward or downward
- Seasonal patterns in discussions
- Correlation between different topics
- Spikes that might correspond to significant events or announcements
