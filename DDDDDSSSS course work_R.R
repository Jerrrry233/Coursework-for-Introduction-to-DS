# Load necessary libraries
library(httr)
library(dplyr)
library(stringr)
library(tibble)
library(purrr)
library(jsonlite)
library(tidytext)
library(SnowballC)
library(textdata)
library(tidyr)


# ==== Gather moon data ====

# These websites are checked and get approval for using from the lecture instructor

# Define the list of URLs for moon phase data
urls <- c(
  "http://maps.seds.org/StarDate/moon2012.txt",
  "http://maps.seds.org/StarDate/moon2013.txt",
  "http://maps.seds.org/StarDate/moon2014.txt",
  "http://maps.seds.org/StarDate/moon2015.txt",
  "http://maps.seds.org/StarDate/moon2016.txt",
  "http://maps.seds.org/StarDate/moon2017.txt",
  "http://maps.seds.org/StarDate/moon2018.txt",
  "http://maps.seds.org/StarDate/moon2019.txt",
  "http://maps.seds.org/StarDate/moon2020.txt",
  "http://maps.seds.org/StarDate/moon2021.txt",
  "http://maps.seds.org/StarDate/moon2022.txt"
)

# Function to scrape moon phase data from a single URL
scrape_moon_data <- function(url) {
  response <- GET(url)
  
  if (response$status_code == 200) {
    raw_text <- content(response, as = "text")
    
    # Extract moon phase and dates using regex
    matches <- str_match_all(
      raw_text, 
      "(New moon|First quarter|Full moon|Last quarter)\\s+JDE.*?(\\d{1,2}-\\d{1,2}-\\d{4})"
    )[[1]]
    
    # Create a tibble with the extracted data
    tibble(
      MoonPhase = matches[, 2],
      Date = as.Date(matches[, 3], format = "%d-%m-%Y")
    )
  } else {
    message("Failed to fetch data from: ", url)
    return(tibble(MoonPhase = character(), Date = as.Date(character())))
  }
}

# Scrape data from all URLs and combine into one dataset
moon_data <- map_dfr(urls, scrape_moon_data)

# Filter to include only rows with dates in 2012–2022
cleaned_moon_data <- moon_data %>%
  filter(Date >= as.Date("2012-01-01") & Date <= as.Date("2022-12-31")) %>%
  distinct(Date, .keep_all = TRUE)  # Remove duplicates if any

# Display the moon dataset
View(cleaned_moon_data)

# ==== Select News data ====

# Load the news data set
news_data <- jsonlite::stream_in(file("C:\\Users\\16476\\OneDrive\\Desktop\\News_Category_Dataset_v3.json"))

# Convert the date column in news_data to the correct format
news_data <- news_data %>%
  mutate(date = as.Date(date, format = "%Y-%m-%d"))

# Match news data with moon dates
matched_news <- news_data %>%
  filter(date %in% cleaned_moon_data$Date) %>%
  inner_join(cleaned_moon_data, by = c("date" = "Date"))

# Flag unmatched moon dates
all_dates <- cleaned_moon_data %>%
  anti_join(news_data, by = c("Date" = "date")) %>%
  mutate(headline = NA, category = NA, short_description = NA)

# Combine matched and unmatched data
news_moon_dataset <- bind_rows(
  matched_news %>%
    select(date, MoonPhase, headline, category, short_description),
  all_dates %>%
    select(date = Date, MoonPhase, headline, category, short_description)
)

# Display the final dataset
View(news_moon_dataset)
str(news_moon_dataset)

# Define the categories to keep
categories_to_keep <- c("POLITICS", "WELLNESS", "ENTERTAINMENT", 
                        "QUEER VOICES", "BUSINESS", "SPORTS")

# Filter the dataset to keep only the specified categories
moon_news_category <- news_moon_dataset %>%
  filter(category %in% categories_to_keep)

# Check the structure of the filtered dataset
View(moon_news_category)
str(moon_news_category)

# ==== Clean the data ====

# Summarise missing values
moon_news_category %>%
  summarise(
    missing_headline = sum(is.na(headline)),
    missing_short_description = sum(is.na(short_description))
  ) %>%
  print()

# Remove rows with NA values
moon_news_cleaned <- moon_news_category %>%
  filter(!is.na(headline) & !is.na(short_description))

# Remove rows with empty strings in critical columns
moon_news_cleaned <- moon_news_cleaned %>%
  filter(headline != "" & short_description != "")

# Verify the cleaned data set
View(moon_news_cleaned)
str(moon_news_cleaned)

# ==== Text Proprocess ====

# Define stop words
data("stop_words")

# Combine Headline and Short Description
moon_news_preprocessed <- moon_news_cleaned %>%
  mutate(
    combined_text = paste(headline, short_description, sep = " ") # Combine headline and short description
  )

# Tokenization
moon_news_preprocessed <- moon_news_preprocessed %>%
  unnest_tokens(word, combined_text) # Tokenize the combined text

# Lowercase Transformation
moon_news_preprocessed <- moon_news_preprocessed %>%
  mutate(word = tolower(word)) # Convert all tokens to lowercase

# Stop-word Removal
moon_news_preprocessed <- moon_news_preprocessed %>%
  anti_join(stop_words, by = "word") # Remove stop words

# Stemming
moon_news_preprocessed <- moon_news_preprocessed %>%
  mutate(word = wordStem(word, language = "en")) # Apply stemming to tokens

# Group back into the dataset
moon_news_preprocessed <- moon_news_preprocessed %>%
  group_by(date, MoonPhase, category, headline, short_description) %>%
  summarise(
    processed_text = paste(word, collapse = " ") # Combine processed tokens back into a single text column
  ) %>%
  ungroup()

# View the preprocessed dataset
View(moon_news_preprocessed)
str(moon_news_preprocessed)

# ==== use TF-IDF in conjunction with Afinn Lexicon for Sentiment Analysis ====

# Assign a unique document identifier
moon_news_preprocessed <- moon_news_preprocessed %>%
  mutate(document_id = row_number())  # Add unique IDs for each document

# Tokenize processed_text and count word frequencies
moon_news_tokens <- moon_news_preprocessed %>%
  unnest_tokens(word, processed_text) %>%
  count(document_id, word, sort = TRUE)  # Count term frequency by document

# Calculate TF-IDF scores
moon_news_tfidf <- moon_news_tokens %>%
  bind_tf_idf(word, document_id, n)  # Compute TF-IDF

# Match tokens with AFINN lexicon and calculate weighted scores
afinn_lexicon <- get_sentiments("afinn")

TFIDF_Moon_News_Sentiment <- moon_news_tfidf %>%
  inner_join(afinn_lexicon, by = "word") %>%
  mutate(weighted_sentiment = tf_idf * value) %>%  # Multiply TF-IDF by sentiment value
  group_by(document_id) %>%  # Aggregate by document
  summarise(
    headline = first(moon_news_preprocessed$headline[document_id]),
    short_description = first(moon_news_preprocessed$short_description[document_id]),
    matched_positive_words = paste(word[value > 0], collapse = ", "),
    matched_negative_words = paste(word[value < 0], collapse = ", "),
    weighted_positive_words = sum(weighted_sentiment[value > 0], na.rm = TRUE),
    weighted_negative_words = sum(weighted_sentiment[value < 0], na.rm = TRUE),
    weighted_sentiment_score = sum(weighted_sentiment, na.rm = TRUE)
  ) %>%
  ungroup()  # Exit grouping

# Filter out rows with no matched words or sentiment scores
TFIDF_Moon_News_Sentiment <- TFIDF_Moon_News_Sentiment %>%
  filter(
    !is.na(weighted_sentiment_score) &
      (weighted_positive_words != 0 | weighted_negative_words != 0)
  )

# Add date, MoonPhase, and category columns from moon_news_preprocessed
TFIDF_Moon_News_Sentiment <- TFIDF_Moon_News_Sentiment %>%
  left_join(moon_news_preprocessed %>% select(document_id, date, MoonPhase, category),
            by = "document_id")

# Reorder columns for readability
TFIDF_Moon_News_Sentiment <- TFIDF_Moon_News_Sentiment %>%
  select(date, MoonPhase, category, everything())  # Move date-related columns to the front


# Preview the updated dataset
View(TFIDF_Moon_News_Sentiment)
write.csv(TFIDF_Moon_News_Sentiment, "C:\\Users\\16476\\OneDrive\\Desktop\\TFIDF_Moon_News_Sentiment.csv", row.names = FALSE)

# ==== Summary statistics ====
summary_stats <- TFIDF_Moon_News_Sentiment %>%
  group_by(MoonPhase, category) %>%
  summarise(
    mean_sentiment = mean(weighted_sentiment_score, na.rm = TRUE),
    sd_sentiment = sd(weighted_sentiment_score, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(desc(mean_sentiment))

# Print the summary statistics
View(summary_stats)

# ==== Visualization: Bar Chart of Mean Sentiment Score by Moon Phase and Category ====
# Create a bar chart
# Define the custom order for Moon Phases
moon_phase_order <- c("New moon", "First quarter", "Full moon", "Last quarter")

# Ensure the MoonPhase column follows the desired order
summary_stats$MoonPhase <- factor(summary_stats$MoonPhase, levels = moon_phase_order)

# ==== Create the bar chart on each moon phase ====
library(ggplot2)

ggplot(summary_stats, aes(x = MoonPhase, y = mean_sentiment, fill = category)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(
    title = "Mean Weighted Sentiment Score by Moon Phase and Category",
    x = "Moon Phase",
    y = "Mean Weighted Sentiment Score",
    fill = "Category"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for readability
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
  )


# ==== grouped bar chart ====
# Ensure the MoonPhase column follows the desired order
summary_stats$MoonPhase <- factor(summary_stats$MoonPhase, levels = moon_phase_order)

# Create the grouped bar chart
ggplot(summary_stats, aes(x = category, y = mean_sentiment, fill = MoonPhase)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("New moon" = "#1f77b4", 
                               "First quarter" = "#2ca02c", 
                               "Full moon" = "#d62728", 
                               "Last quarter" = "#ff7f0e")) +
  labs(
    title = "Mean Weighted Sentiment Score by News Category and Moon Phase",
    x = "News Category",
    y = "Mean Weighted Sentiment Score",
    fill = "Moon Phase"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for readability
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
  )









# ==== test for Normality ====
# Load required libraries
# Group data by Moon Phase and test for normality
normality_results <- TFIDF_Moon_News_Sentiment %>%
  group_by(MoonPhase) %>% 
  summarise(
    n = n(),  # Count of samples
    shapiro_p_value = ifelse(n >= 3, 
                             shapiro.test(weighted_sentiment_score)$p.value, 
                             NA)  # Apply Shapiro-Wilk test only if sample size >= 3
  ) %>%
  mutate(
    normality = ifelse(!is.na(shapiro_p_value) & shapiro_p_value > 0.05, 
                       "Normal", 
                       "Not Normal")  # Interpret results
  )

# View the summary table
print(normality_results)
