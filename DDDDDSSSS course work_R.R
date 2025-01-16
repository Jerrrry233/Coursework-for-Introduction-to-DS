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
# Load necessary library
library(dplyr)

# Load your dataframe
# Assuming the dataframe is named `TFIDF_Moon_News_Sentiment`

# Generate attribute table
attribute_table <- data.frame(
  Label = c(
    "date", 
    "MoonPhase", 
    "category", 
    "matched_positive_words", 
    "matched_negative_words", 
    "weighted_positive_words", 
    "weighted_negative_words", 
    "weighted_sentiment_score"
  ),
  Data_Category = c(
    "Date", 
    "Categorical", 
    "Categorical", 
    "Text", 
    "Text", 
    "Numeric", 
    "Numeric", 
    "Numeric"
  ),
  Summary = c(
    paste("Range:", min(TFIDF_Moon_News_Sentiment$date), "to", max(TFIDF_Moon_News_Sentiment$date)),
    paste("Unique Phases:", paste(unique(TFIDF_Moon_News_Sentiment$MoonPhase), collapse = ", ")),
    paste("Unique Categories:", paste(unique(TFIDF_Moon_News_Sentiment$category), collapse = ", ")),
    paste("Text data with", nrow(TFIDF_Moon_News_Sentiment), "entries"),
    paste("Text data with", nrow(TFIDF_Moon_News_Sentiment), "entries"),
    paste(
      "Min:", min(TFIDF_Moon_News_Sentiment$weighted_positive_words, na.rm = TRUE),
      "Max:", max(TFIDF_Moon_News_Sentiment$weighted_positive_words, na.rm = TRUE),
      "Mean:", mean(TFIDF_Moon_News_Sentiment$weighted_positive_words, na.rm = TRUE)
    ),
    paste(
      "Min:", min(TFIDF_Moon_News_Sentiment$weighted_negative_words, na.rm = TRUE),
      "Max:", max(TFIDF_Moon_News_Sentiment$weighted_negative_words, na.rm = TRUE),
      "Mean:", mean(TFIDF_Moon_News_Sentiment$weighted_negative_words, na.rm = TRUE)
    ),
    paste(
      "Min:", min(TFIDF_Moon_News_Sentiment$weighted_sentiment_score, na.rm = TRUE),
      "Max:", max(TFIDF_Moon_News_Sentiment$weighted_sentiment_score, na.rm = TRUE),
      "Mean:", mean(TFIDF_Moon_News_Sentiment$weighted_sentiment_score, na.rm = TRUE)
    )
  )
)

# Display the attribute table
View(attribute_table)


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


# ==== Grouped Bar Chart with Sentiment Scores ====

summary_stats$MoonPhase <- factor(summary_stats$MoonPhase, levels = c("New moon", "First quarter", "Full moon", "Last quarter"))

ggplot(summary_stats, aes(x = category, y = mean_sentiment, fill = MoonPhase)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +  # Adjust bar width and dodge for spacing
  geom_text(aes(label = round(mean_sentiment, 2)),  # Add labels to bars
            position = position_dodge(width = 0.8), 
            vjust = -0.5, 
            size = 3.5) +  # Adjust text size
  scale_fill_manual(values = c("New moon" = "#1e90ff",    # Midnight blue for New moon
                               "First quarter" = "#32cd32", # Lime green for First quarter
                               "Full moon" = "#ff4500",     # Orange red for Full moon
                               "Last quarter" = "#ffcc00")) # Golden yellow for Last quarter
labs(
  title = "Mean Weighted Sentiment Score by News Category and Moon Phase",
  x = "News Category",
  y = "Mean Weighted Sentiment Score",
  fill = "Moon Phase"
) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels for readability
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    legend.position = "top"  # Place legend on top for better aesthetics
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

# ==== test for Normality among News Categories====
# Create a function to perform the Shapiro-Wilk test for each category and moon phase
test_normality <- function(data) {
  data %>%
    group_by(category, MoonPhase) %>%
    summarise(
      n = n(),  # Sample size
      shapiro_p_value = ifelse(n > 3, shapiro.test(weighted_sentiment_score)$p.value, NA),
      normality = ifelse(shapiro_p_value > 0.05, "Normal", "Not Normal")
    ) %>%
    ungroup()
}

# Apply the function to the dataset
normality_results <- test_normality(TFIDF_Moon_News_Sentiment)

# Print the results
View(normality_results)


# Perform Descriptive Statistics
descriptive_stats <- TFIDF_Moon_News_Sentiment %>%
  group_by(category, MoonPhase) %>%
  summarise(
    N = n(),
    mean_sentiment = mean(weighted_sentiment_score, na.rm = TRUE),
    sd_sentiment = sd(weighted_sentiment_score, na.rm = TRUE),
    min_sentiment = min(weighted_sentiment_score, na.rm = TRUE),
    max_sentiment = max(weighted_sentiment_score, na.rm = TRUE)
  ) %>%
  ungroup()

# View descriptive statistics
View(descriptive_stats)

# ==== Perform Kruskal-Wallis Test for each category ====

# Perform Kruskal-Wallis Test for each category
kruskal_results <- TFIDF_Moon_News_Sentiment %>%
  group_by(category) %>%
  summarise(
    test = list(kruskal.test(weighted_sentiment_score ~ MoonPhase, data = pick(everything()))),
    .groups = 'drop'
  )

# Extract results for each category
kruskal_summary <- kruskal_results %>%
  mutate(
    statistic = sapply(test, function(x) x$statistic),
    df = sapply(test, function(x) x$parameter),
    p_value = sapply(test, function(x) x$p.value)
  ) %>%
  select(category, statistic, df, p_value)

# View the summary results
print(kruskal_summary)

# ==== To find out where the significant differences are ====
# Filter the data for ENTERTAINMENT and WELLNESS categories
filtered_data <- TFIDF_Moon_News_Sentiment %>%
  filter(category %in% c("ENTERTAINMENT", "WELLNESS"))

# Perform Kruskal-Wallis test for each category
kruskal_results <- list()
for (cat in unique(filtered_data$category)) {
  category_data <- filtered_data %>% filter(category == cat)
  kruskal_results[[cat]] <- kruskal.test(weighted_sentiment_score ~ MoonPhase, data = category_data)
  print(paste("Kruskal-Wallis Test for category:", cat))
  print(kruskal_results[[cat]])
}

# Post-hoc pairwise comparisons using Dunn test
dunn_results <- list()
for (cat in unique(filtered_data$category)) {
  category_data <- filtered_data %>% filter(category == cat)
  dunn_results[[cat]] <- dunn.test(category_data$weighted_sentiment_score, category_data$MoonPhase, method = "bonferroni")
  print(paste("Dunn Test for category:", cat))
  print(dunn_results[[cat]])
}

# Print pairwise comparison results directly without creating a table
print_pairwise_comparisons <- function(dunn_result, category) {
  if (is.null(dunn_result$comparisons) || length(dunn_result$comparisons) == 0) {
    warning(paste("No pairwise comparisons available for category:", category))
    return()
  }
  
  cat("\nPairwise Comparisons for Category:", category, "\n")
  comparisons <- dunn_result$comparisons
  Z_scores <- round(as.numeric(dunn_result$Z), 2)
  P_values <- round(as.numeric(dunn_result$P.unadj), 3)
  Adjusted_P_values <- round(as.numeric(dunn_result$P.adj), 3)
  
  for (i in seq_along(comparisons)) {
    cat(comparisons[i], "\n")
    cat("  Test Statistic (Z):", Z_scores[i], "\n")
    cat("  P-value:", P_values[i], "\n")
    cat("  Adjusted P-value:", Adjusted_P_values[i], "\n\n")
  }
}

# Print pairwise comparisons for ENTERTAINMENT and WELLNESS
if (!is.null(dunn_results$ENTERTAINMENT)) {
  print_pairwise_comparisons(dunn_results$ENTERTAINMENT, "ENTERTAINMENT")
}

if (!is.null(dunn_results$WELLNESS)) {
  print_pairwise_comparisons(dunn_results$WELLNESS, "WELLNESS")
}

# Generate a boxplot for each category with adjusted y-axis limits
ggplot(filtered_data, aes(x = MoonPhase, y = weighted_sentiment_score, fill = MoonPhase)) +
  geom_boxplot() +
  facet_wrap(~ category, scales = "free") +
  labs(
    title = "Boxplot of Sentiment Scores by Moon Phase",
    y = "Weighted Sentiment Score",
    x = "Moon Phase"
  ) +
  scale_y_continuous(
    limits = c(-4, 4), 
    breaks = seq(-2, 2, by = 0.5)
  ) +
  theme_minimal()

create_pairwise_network <- function(dunn_results, category) {
  comparison_results <- dunn_results[[category]]
  pairwise_ranks <- data.frame(
    MoonPhase1 = gsub(" - .*", "", comparison_results$comparisons),
    MoonPhase2 = gsub(".* - ", "", comparison_results$comparisons),
    RankMeanDifference = comparison_results$Z,
    p_value = comparison_results$P.adj
  )
  
  # Highlight significant relationships (p-value < 0.05)
  pairwise_ranks$edge_color <- ifelse(pairwise_ranks$p_value < 0.05, "darkred", "gray")
  pairwise_ranks$edge_width <- ifelse(pairwise_ranks$p_value < 0.05, 2, 1)
  
  graph <- graph_from_data_frame(pairwise_ranks, directed = FALSE)
  
  plot(graph,
       edge.width = pairwise_ranks$edge_width,
       edge.color = pairwise_ranks$edge_color,
       edge.label = round(pairwise_ranks$RankMeanDifference, 2),
       vertex.color = "skyblue",
       vertex.size = 30,
       vertex.label.cex = 0.8,
       main = paste("Pairwise Comparisons -", category)
  )
}

# Create networks for ENTERTAINMENT and WELLNESS
create_pairwise_network(dunn_results, "ENTERTAINMENT")
create_pairwise_network(dunn_results, "WELLNESS")


# ==== word cloud ====
# Install and load required libraries
library(wordcloud)
# Data preparation: Combine matched_positive_words and matched_negative_words
data_combined <- TFIDF_Moon_News_Sentiment %>%
  select(MoonPhase, matched_positive_words, matched_negative_words) %>%
  mutate(
    all_words = paste(matched_positive_words, matched_negative_words, sep = " ")
  ) %>%
  separate_rows(all_words, sep = " ") %>%
  filter(all_words != "") %>%
  mutate(
    sentiment = case_when(
      all_words %in% unlist(strsplit(TFIDF_Moon_News_Sentiment$matched_positive_words, " ")) ~ "positive",
      all_words %in% unlist(strsplit(TFIDF_Moon_News_Sentiment$matched_negative_words, " ")) ~ "negative",
      TRUE ~ NA_character_
    )
  ) %>%
  na.omit()

# Calculate word frequency by moon phase and sentiment
word_freq <- data_combined %>%
  group_by(MoonPhase, all_words, sentiment) %>%
  summarise(freq = n(), .groups = "drop") %>%
  arrange(desc(freq))

# Generate word clouds
par(mfrow = c(2, 2))  # Arrange 4 plots in a 2x2 grid
moon_sizes <- c("New moon" = 3, "First quarter" = 4, "Full moon" = 6, "Last quarter" = 5)  # Set moon phase sizes

for (phase in unique(word_freq$MoonPhase)) {
  phase_data <- word_freq %>% filter(MoonPhase == phase)
  
  # Separate positive and negative words for coloring
  positive_words <- phase_data %>% filter(sentiment == "positive")
  negative_words <- phase_data %>% filter(sentiment == "negative")
  
  # Generate the word cloud for the current moon phase
  wordcloud(
    words = c(positive_words$all_words, negative_words$all_words),
    freq = c(positive_words$freq, negative_words$freq),
    colors = c(rep("pink", nrow(positive_words)), rep("grey", nrow(negative_words))),
    max.words = 20,
    scale = c(moon_sizes[phase], moon_sizes[phase] - 1),
    main = paste("Word Cloud for", phase)
  )
}

par(mfrow = c(1, 1))  # Reset plotting layout

# ==== heatmap ====
install.packages("lubridate")
library(lubridate)
# Create a year column for aggregation
TFIDF_Moon_News_Sentiment <- TFIDF_Moon_News_Sentiment %>%
  mutate(year = format(date, "%Y"))

# Calculate mean sentiment score by year and MoonPhase
heatmap_data <- TFIDF_Moon_News_Sentiment %>%
  group_by(year, MoonPhase) %>%
  summarise(mean_sentiment = mean(weighted_sentiment_score, na.rm = TRUE), .groups = 'drop')

# Ensure MoonPhase is ordered
heatmap_data$MoonPhase <- factor(heatmap_data$MoonPhase, levels = c("New moon", "First quarter", "Full moon", "Last quarter"))

# Create the heatmap
ggplot(heatmap_data, aes(x = year, y = MoonPhase, fill = mean_sentiment)) +
  geom_tile(color = "white") +  # Add white grid lines for clarity
  scale_fill_gradient2(
    low = "blue",
    mid = "white",
    high = "red",  # Vivid red for higher scores
    midpoint = 0,
    name = "Sentiment Score"
  ) +
  labs(
    title = "Sentiment Score Heatmap Across Moon Phases by Year",
    x = "Year",
    y = "Moon Phase"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 0.5, size = 10),  # Keep x-axis labels horizontal
    axis.text.y = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14)
  )

# ==== Violin plot ====
# Ensure the MoonPhase column follows the desired order
TFIDF_Moon_News_Sentiment$MoonPhase <- factor(
  TFIDF_Moon_News_Sentiment$MoonPhase, 
  levels = c("New moon", "First quarter", "Full moon", "Last quarter")
)

# Create the faceted violin plot
violin_plot <- ggplot(TFIDF_Moon_News_Sentiment, 
                      aes(x = MoonPhase, y = weighted_sentiment_score, fill = MoonPhase)) +
  geom_violin(trim = FALSE, alpha = 0.8) +
  geom_boxplot(width = 0.1, position = position_dodge(width = 0.9), alpha = 0.7, outlier.shape = NA) +
  facet_wrap(~category, scales = "free_y") +
  scale_fill_manual(values = c(
    "New moon" = "#1f77b4", 
    "First quarter" = "#2ca02c", 
    "Full moon" = "#d62728", 
    "Last quarter" = "#9467bd"
  )) +
  labs(
    title = "Distribution of Sentiment Scores by Moon Phase and News Category",
    x = "Moon Phase",
    y = "Weighted Sentiment Score",
    fill = "Moon Phase"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

# Print the plot
print(violin_plot)








