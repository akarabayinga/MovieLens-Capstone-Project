###################################################################################
# Movie Recommendation System
# HarvardX Data Science Professional Certificate
# - PH125.9x Data Science: Capstone (MovieLens Project)
# Author: Audrey Karabayinga
###################################################################################

## ----setup, include = FALSE------------------------------------------------------------------------------------------------------------

# Load setup
knitr::opts_chunk$set(
  echo = FALSE,
  message = FALSE,
  warning = FALSE
)

## ---- Create edx, set and validation set (final hold-out test set), cache=TRUE---------------------------------------------------------

###################################################################################
# Download MovieLens 10M data, create *edx* and *validation* data sets
###################################################################################

# Use code provided by course instructor to download MovieLens 10M data and create edx data set (90% of data) and validation data set (10% of data)

if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(
  text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
  col.names = c("userId", "movieId", "rating", "timestamp")
)

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(
  movieId = as.numeric(movieId),
  title = as.character(title),
  genres = as.character(genres)
)

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = "Rounding") # Set.seed for reproducibility
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


## ---- R packages-----------------------------------------------------------------------------------------------------------------------

###################################################################################
# Install other required packages and load libraries
###################################################################################

if (!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if (!require(tinytex)) install.packages("tinytex", repos = "http://cran.us.r-project.org")
if (!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if (!require(yaml)) install.packages("yaml", repos = "http://cran.us.r-project.org")
if (!require(naniar)) install.packages("naniar", repos = "http://cran.us.r-project.org")
if (!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if (!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if (!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if (!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")
if (!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if (!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
if (!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")

library(kableExtra)
library(tinytex)
library(knitr)
library(yaml)
library(naniar)
library(gridExtra)
library(tidyverse)
library(caret)
library(rmarkdown)
library(lubridate)
library(scales)
library(grid)


## ---- Custom theme, color palette, include = FALSE-------------------------------------------------------------------------------------

###################################################################################
# Create custom plot theme for all plots in the report
###################################################################################

theme_1 <- function() {
  font <- "sans" # assign family font
  theme_classic() %+replace% # modify from base theme
    theme(
      plot.title = element_text(
        family = font,
        size = 10,
        face = "bold",
        hjust = 0,
        vjust = 2
      ),
      axis.title = element_text(
        family = font,
        size = 8
      ),
      axis.text = element_text(
        family = font,
        size = 7
      ),
      legend.position = "top",
      legend.key.size = unit(0.35, "cm"),
      legend.title = element_text(
        size = 8
      ),
      plot.caption = element_text(
        size = 6,
        hjust = 1
      )
    )
}


## ---- data set-dimensions--------------------------------------------------------------------------------------------------------------

# Explore data
###################################################################################

# View dimensions for edx and validation data sets
edx_dim <- dim(edx)
validation_dim <- dim(validation)


## ---- data-figure, fig.cap = "Allocation of data", fig.height = 1, fig.width = 2.5-----------------------------------------------------

knitr::include_graphics("data_segments.png")


## ---- structure------------------------------------------------------------------------------------------------------------------------

# Create table showing edx classes and the first 10 observations
rbind(
  lapply(edx, class),
  head(edx, 10)
) %>%
  kbl(
    caption = "Overview of edx data set",
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped", "scale_down"),
    font_size = 8
  ) %>%
  row_spec(0, bold = T) %>%
  pack_rows("Class", 1, 1) %>%
  pack_rows("First 10 observations", 2, 11)


## ---- Missing values - edx, include = FALSE--------------------------------------------------------------------------------------------

# Check whether edx data set has any missing values
gg_miss_var(edx) +
  theme_1() +
  labs(
    title = "Missing values in edx data set",
    x = "Variables",
    y = "Number or missing values"
  )


## ---- Sparsity-------------------------------------------------------------------------------------------------------------------------

# All users did not rate all movies - compute the sparsity of the user-movie matrix

# Compute nonzero entries (i.e., total number of ratings in edx)
nonzero_entries <- nrow(edx)

# Compute total entries (i.e., total ratings if all users had rated all movies)

# Compute number of distinct users
distinct_users <- edx %>%
  distinct(userId) %>%
  nrow()

# Compute number of distinct movies
distinct_movies <- edx %>%
  distinct(movieId) %>%
  nrow()
# Compute total entries
tot_entries <- distinct_movies * distinct_users

# Compute sparsity
sparsity <- 1 - (nonzero_entries / tot_entries)


## ---- Sparsity-matrix, fig.cap = "User-movie matrix containing 50 movies and users from edx", fig.height = 4,  fig.width = 4-----------
  
# Create user movie matrix (for 50 users and 50 movies) showing whether a user in a certain row rated a movie in a certain column, to show how sparse this matrix is.

# Select a sample of 50 users
set.seed(1, sample.kind = "Rounding") # Set.seed for reproducibility
users <- sample(unique(edx$userId), 50)

# Select the movieIDs and ratings for all 50 users from edx
edx %>%
  filter(userId %in% users) %>%
  select(userId, movieId, rating) %>%
  mutate(rating = 1) %>%
  pivot_wider(
    names_from = movieId,
    values_from = rating
  ) %>%
  select(userId, sample(ncol(.), 50)) %>%
  # selecta sample of 50 movies
  pivot_longer(-1,
    names_to = "movieId",
    values_to = "rating"
  ) %>%
  mutate(
    rating = replace_na(rating, 0),
    rating = ifelse(rating == 1, "Yes", "No")
  ) %>%
  ggplot(aes(as.factor(movieId), as.factor(userId))) +
  geom_tile(aes(fill = rating),
                color = "grey50",
    alpha = 0.5
  ) +
  scale_fill_manual(values = c("white", "#205493")) +
  labs(
    caption = "Source: MovieLens 10M Dataset",
    x = "Movies",
    y = "Users",
    fill = "Movie rated?"
  ) +
  scale_x_discrete(guide = guide_axis(angle = 90)) +
  theme_1() +
  theme(
    axis.text.x = element_text(size = 5),
    axis.text.y = element_text(size = 5)
  )


## ---- Rating-summary-------------------------------------------------------------------------------------------------------------------

# Explore *rating* column
###################################################################################

# A. Summary for *rating* column:

# Summarise rating column (include mean, sd, 1st & 3rd quartile)
summ_rating <- edx %>%
  summarise(
    min = min(rating),
    mean = round(mean(rating), digits = 2),
    sd = round(sd(rating), digits = 2),
    q_1st = quantile(rating, probs = 25 / 100),
    median = median(rating),
    q_3rd = quantile(rating, probs = 75 / 100),
    max = max(rating)
  )


## ---- Rating-scores, fig.cap = "Distribution of rating scores ", fig.height = 3.5------------------------------------------------------

# B. Distribution of *rating* column:

# Plot distribution of rating scores
edx %>%
  group_by(rating) %>%
  summarise(
    n = n(),
    star = ifelse(rating %in% c(1, 2, 3, 4, 5),
      "Whole star", "Half star"
    )
  ) %>%
  # Remove duplicates
  filter(!duplicated(rating)) %>%
  ungroup() %>%
  ggplot(aes(rating, n, fill = star)) +
  geom_col(alpha = 0.5) +
  labs(
    caption = "Source: MovieLens 10M data set",
    x = "Rating score",
    y = "Number of ratings",
    fill = "Type"
  ) +
  # Show number of ratings per score
  geom_text(aes(
    label = format(n, big.mark = ","),
    vjust = -0.5
  ),
  size = 2.5,
  alpha = 0.5
  ) +
  scale_x_continuous(breaks = breaks_pretty(n = 10)) +
  scale_y_continuous(
    label = scales::comma,
    breaks = breaks_pretty(n = 6)
  ) +
  # Add line showing median rating score
  geom_vline(
    xintercept = summ_rating$median,
    linetype = "dashed",
    alpha = 0.5,
    color = "black"
  ) +
  geom_label(aes(4.25, 3000000),
    label = paste0("Median = ", summ_rating$median),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Add standard deviation of rating scores
  annotate("text",
    x = 1.5,
    y = 2000000,
    size = 2,
    color = "grey50",
    label = str_c(
      "1st Quartile = ", summ_rating$q_1st,
      "\n Mean = ", summ_rating$mean,
      "\n Std. Dev. = ", summ_rating$sd,
      "\n 3rd Quartile = ", summ_rating$q_3rd
    )
  ) +
  scale_fill_manual(values = c("#651D32", "#205493")) +
  theme_1() +
  theme(
    legend.key.size = unit(0.5, "cm")
  )


## ---- Distinct-movies------------------------------------------------------------------------------------------------------------------

# Explore *movieId* column
###################################################################################

# A. Summary for *movieId* column:

# Number of ratings

# Summarise number of ratings per movie (min, mean, sd, median, max)
summ_movie_rating <- edx %>%
  group_by(movieId) %>%
  summarise(n = n()) %>%
  summarise(
    min = min(n),
    mean = mean(n),
    sd = sd(n),
    median = median(n),
    max = max(n)
  )

# Rating scores

# Summarise average rating score per movie (min, mean, sd, median, max)
summ_movie_score <- edx %>%
  group_by(movieId) %>%
  summarise(avg_score = mean(rating)) %>%
  summarise(
    min = min(avg_score),
    mean = mean(avg_score),
    sd = sd(avg_score),
    median = median(avg_score),
    max = max(avg_score)
  )

# B. Compute coefficient of variation (CV)

# Number of ratings per movie
cv_movie_rating <- summ_movie_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# Average rating score per movie
cv_movie_score <- summ_movie_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)


## ---- Movie-column, fig.cap = " Distribution of number of ratings (A) and average rating scores (B) across movies", fig.height = 3.5----

# C. Distribution for *movieId* column:

# Number of ratings

# Plot distribution of number of ratings across movies
ratings_m <- edx %>%
  group_by(movieId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median rating per movie
  geom_vline(
    xintercept = summ_movie_rating$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(700, 475),
    label = paste0("Median = ", round(summ_movie_rating$median, 0)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 10000,
    y = 350,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", summ_movie_rating$min,
      "\n Mean = ", round(summ_movie_rating$mean, 0),
      "\n Std. Dev. = ", format(round(summ_movie_rating$sd, 0),
        big.mark = ","
      ),
      "\n Maximum = ", format(summ_movie_rating$max,
        big.mark = ","
      )
    )
  ) +
  scale_x_log10(label = scales::comma) +
  labs(
    title = "A",
    x = "Number of ratings",
    y = "Number of movies"
  ) +
  theme_1()

# Rating scores

# Plot distribution of average rating scores across movies
scores_m <- edx %>%
  group_by(movieId) %>%
  summarise(avg_score = mean(rating)) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median average rating score per movie
  geom_vline(
    xintercept = summ_movie_score$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(2.7, 850),
    label = paste0("Median = ", round(summ_movie_score$median, 2)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 1.5,
    y = 600,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", summ_movie_score$min,
      "\n Mean = ", round(summ_movie_score$mean, 2),
      "\n Std. Dev. = ", round(summ_movie_score$sd, 3),
      "\n Maximum = ", summ_movie_score$max
    )
  ) +
  labs(
    title = "B",
    x = "Average rating score",
    y = "Number of movies"
  ) +
  theme_1()

# Plot the 2 graphs on one layout
grid.arrange(ratings_m, scores_m,
  ncol = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- Movies-Top-Bottom----------------------------------------------------------------------------------------------------------------

# D. Best and worst movies - number of ratings / average rating scores:

# Top and bottom 5 movies by number of ratings and their corresponding average rating scores

# Top 5 most rated movies - number of ratings
top_5 <- edx %>%
  group_by(movieId) %>%
  summarise(
    title = title,
    num = n(),
    avg_rating = mean(rating)
  ) %>%
  distinct(movieId, .keep_all = TRUE) %>%
  ungroup() %>%
  slice_max(
    order_by = num,
    n = 5
  )

# 5 of the least rated movies - number of ratings
bottom_5 <- edx %>%
  group_by(movieId) %>%
  summarise(
    title = title,
    num = n(),
    avg_rating = mean(rating)
  ) %>%
  distinct(movieId, .keep_all = TRUE) %>%
  ungroup() %>%
  slice_min(
    order_by = num,
    n = 5 # returns all 126 movies which were rated only once
  ) %>%
  slice_sample(n = 5) # randomly select only five of these least rated movies

# Combine the top and bottom movies into a table with number of ratings and average rating scores for each
rbind(top_5, bottom_5) %>%
  mutate(
    num = format(num, big.mark = ","),
    avg_rating = round(avg_rating, 2)
  ) %>%
  select(-movieId) %>%
  kbl(
    caption = "Top and bottom movies by number of ratings",
    col.names = c("Title (Release year)", "Number of ratings", "Average rating score"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped"),
    font_size = 8
  ) %>%
  row_spec(0, bold = T) %>%
  pack_rows("Most rated", 1, 5) %>%
  pack_rows("Least rated", 6, 10) %>%
  footnote(symbol = "The edx data set contained 126 movies rated by a single user (i.e., 'least rated'). Only 5 of these are shown in this table.")


## ---- Distinct-users-------------------------------------------------------------------------------------------------------------------

# Explore *userId* column
###################################################################################

# A. Summary for *userId* column:

# Number of ratings

# Summarise number of ratings per user (min, mean, sd, median, max)
summ_user_rating <- edx %>%
  group_by(userId) %>%
  summarise(n = n()) %>%
  summarise(
    min = min(n),
    mean = mean(n),
    sd = sd(n),
    median = median(n),
    max = max(n)
  )

# Rating scores

# Summarise average rating score per user (min, mean, sd, median, max)
summ_user_score <- edx %>%
  group_by(userId) %>%
  summarise(avg_score = mean(rating)) %>%
  summarise(
    min = min(avg_score),
    mean = mean(avg_score),
    sd = sd(avg_score),
    median = median(avg_score),
    max = max(avg_score)
  )

# B. Compute coefficient of variation (CV)

# Number of ratings per user
cv_user_rating <- summ_user_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# Average rating score per user
cv_user_score <- summ_user_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(3)


## ---- User-column, fig.cap = "Distribution of number of ratings (A) and average rating scores (B) across users", fig.height = 3.5------

# C. Distribution for *userId* column:

# Number of ratings

# Plot distribution of number of ratings across users
ratings_u <- edx %>%
  group_by(userId) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median
  geom_vline(
    xintercept = summ_user_rating$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(140, 7000),
    label = paste0("Median = ", round(summ_user_rating$median, 0)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 1000,
    y = 5000,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", summ_user_rating$min,
      "\n Mean = ", round(summ_user_rating$mean, 0),
      "\n Std. Dev. = ", round(summ_user_rating$sd, 0),
      "\n Maximum = ", format(summ_user_rating$max,
        big.mark = ","
      )
    )
  ) +
  scale_x_log10(label = scales::comma) +
  labs(
    title = "A",
    caption = "",
    x = "Number of ratings",
    y = "Number of users"
  ) +
  theme_1()

# Rating scores

# Plot distribution of average rating scores across users
scores_u <- edx %>%
  group_by(userId) %>%
  summarise(avg_rating = mean(rating)) %>%
  ggplot(aes(avg_rating)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median
  geom_vline(
    xintercept = summ_user_score$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(3.1, 7000),
    label = paste0("Median: ", round(summ_user_score$median, 2)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 1.5,
    y = 5000,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", summ_user_score$min,
      "\n Mean = ", round(summ_user_score$mean, 2),
      "\n Std. Dev. = ", round(summ_user_score$sd, 3),
      "\n Maximum = ", summ_user_score$max
    )
  ) +
  labs(
    title = "B",
    x = "Average rating score",
    y = "Number of users"
  ) +
  theme_1()

# Plot the 2 graphs on one layout
grid.arrange(ratings_u, scores_u,
  ncol = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- Users-Top-Bottom-----------------------------------------------------------------------------------------------------------------

# D. Top and bottom users - number of ratings / average rating scores:

# Top 5 users by number of ratings
top_5 <- edx %>%
  group_by(userId) %>%
  summarise(
    num = n(),
    avg_rating = mean(rating)
  ) %>%
  distinct(userId, .keep_all = TRUE) %>%
  ungroup() %>%
  slice_max(
    order_by = num,
    n = 5
  )

# Bottom users by number of ratings
bottom_5 <- edx %>%
  group_by(userId) %>%
  summarise(
    num = n(),
    avg_rating = mean(rating)
  ) %>%
  distinct(userId, .keep_all = TRUE) %>%
  ungroup() %>%
  slice_min(
    order_by = num,
    n = 5
  ) %>% # Results in all 28 users with the lowest number of ratings
  slice(1:5) # Select a sample of 5 from users with lowest number of ratings

# Combine the top and bottom users and create a table
rbind(top_5, bottom_5) %>%
  mutate(
    num = format(num, big.mark = ","),
    avg_rating = round(avg_rating, 2)
  ) %>%
  kbl(
    caption = "Top and bottom users by number of ratings",
    col.names = c("User ID", "Number of ratings", "Average rating score"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped"),
    font_size = 8
  ) %>%
  row_spec(0, bold = T) %>%
  pack_rows("Most frequent raters", 1, 5) %>%
  pack_rows("Least frequent raters", 6, 10)


## ---- Original-categories, fig.cap = "Distributions of number of ratings (A) and average rating scores (B) across original genre categories", fig.height = 3.5----

# Explore *genres* column
###################################################################################

# A. Explore original genre categories (797 categories)

# 1. Summary

# Summarise number of ratings per original genre category
g_avg_rating <- edx %>%
  group_by(genres) %>%
  summarise(n = n()) %>%
  summarise(
    min = min(n),
    mean = mean(n),
    sd = sd(n),
    median = median(n),
    max = max(n)
  )

# Summarise average rating score per original genre category
g_avg_score <- edx %>%
  group_by(genres) %>%
  summarise(avg_score = mean(rating)) %>%
  summarise(
    min = min(avg_score),
    mean = mean(avg_score),
    sd = sd(avg_score),
    median = median(avg_score),
    max = max(avg_score)
  )

# 2. Compute coefficient of variation (CV)

# CV for number of ratings per original genre category
cv_g_rating <- g_avg_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# CV for average rating score per original genre category
cv_g_score <- g_avg_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(3)

# 3. Distributions for number of ratings and average rating score

# Plot distribution of number of ratings across original genre categories
ratings_g <- edx %>%
  group_by(genres) %>%
  summarise(n = n()) %>%
  ggplot(aes(n)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median rating per original genre category
  geom_vline(
    xintercept = g_avg_rating$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(275, 40),
    label = paste0("Median = ", round(g_avg_rating$median, 0)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 90,
    y = 30,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", g_avg_rating$min,
      "\n Mean = ", round(g_avg_rating$mean, 0),
      "\n Std. Dev. = ", format(round(g_avg_rating$sd, 0),
        big.mark = ","
      ),
      "\n Maximum = ", format(g_avg_rating$max,
        big.mark = ","
      )
    )
  ) +
  scale_x_log10(label = scales::comma) +
  labs(
    title = "A",
    x = "Number of ratings",
    y = "Number of original genre categories"
  ) +
  theme_1()

# Plot distribution of average rating scores across original genre categories
scores_g <- edx %>%
  group_by(genres) %>%
  summarise(avg_score = mean(rating)) %>%
  ggplot(aes(avg_score)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_density(aes(y = ..count.. * 0.1),
    color = "#651D32"
  ) +
  # Add line showing median average rating score per original genre category
  geom_vline(
    xintercept = g_avg_score$median,
    linetype = "dashed",
    color = "black"
  ) +
  geom_label(aes(3, 80),
    label = paste0("Median = ", round(g_avg_score$median, 2)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  # Show min, mean, standard deviation & max
  annotate("text",
    x = 2,
    y = 40,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", round(g_avg_score$min, 2),
      "\n Mean = ", round(g_avg_score$mean, 2),
      "\n Std. Dev. = ", round(g_avg_score$sd, 3),
      "\n Maximum = ", round(g_avg_score$max, 2)
    )
  ) +
  labs(
    title = "B",
    x = "Average rating score",
    y = "Number of original genre categories"
  ) +
  theme_1()

# Plot the 2 graphs on one layout
grid.arrange(ratings_g, scores_g,
  ncol = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- Number-of-genres-categories, fig.cap = "Distributions of number of movies (A) and number of ratings and average rating scores (B) across number-of-genres categories", fig.height = 3.5----

# B. Explore number-of-genre categories

# Create modified edx dataset with a column (n_genres) for categories based on the number of genres a movie is categorized under
edx_n_genres <- edx %>%
  filter(!genres == "(no genres listed)") %>% # Remove 'no genres listed' category
  mutate(n_genres = paste0(str_count(genres, "\\|") + 1, "-genre"))

# 1. Summary

# Summarise number of ratings per number-of-genre category (1-genre, 2-genre, etc)
gc_avg_rating <- edx_n_genres %>%
  group_by(n_genres) %>%
  summarise(n = n()) %>%
  summarise(
    mean = mean(n),
    sd = sd(n)
  )

# Summarise average rating score per number-of-genre category (1-genre, 2-genre, etc)
gc_avg_score <- edx_n_genres %>%
  group_by(n_genres) %>%
  summarise(avg_score = mean(rating)) %>%
  summarise(
    mean = mean(avg_score),
    sd = sd(avg_score)
  )

# 2. Compute coefficient of variation (CV)

# Number of ratings per number-of-genre category (1-genre, 2-genre, etc)
cv_gc_rating <- gc_avg_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# Average rating score per number-of-genre category (1-genre, 2-genre, etc)
cv_gc_score <- gc_avg_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(4)

# 3. Distributions for number of ratings and average rating score

# Plot distribution of movies across number-of-genre categories
gc_movies <- edx_n_genres %>%
  group_by(n_genres) %>%
  distinct(movieId, .keep_all = TRUE) %>%
  summarise(n = n()) %>%
  ggplot(aes(
    n,
    reorder(n_genres, n,
      decreasing = FALSE
    )
  )) +
  geom_col(
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_text(aes(
    label = format(n, big.mark = ","),
    hjust = -0.25
  ),
  size = 2.5
  ) +
  labs(
    title = "A",
    x = "Number of movies",
    y = "Number of genres per category"
  ) +
  scale_y_discrete() +
  scale_x_continuous(
    label = scales::comma,
    limits = c(0, 5000)
  ) +
  theme_1()

# Plot distribution of number of ratings and average rating scores across number-of-genre categories
gc_ratings <- edx_n_genres %>%
  group_by(n_genres) %>%
  summarise(
    n = n(),
    avg_score = mean(rating)
  ) %>%
  ggplot() +
  geom_col(aes(
    x = n,
    y = reorder(n_genres, n,
      decreasing = FALSE
    )
  ),
  fill = "#205493",
  color = "white",
  alpha = 0.5
  ) +
  geom_point(aes(
    x = avg_score * 600000,
    y = n_genres
  ),
  color = "#651D32"
  ) +
  labs(
    title = "B"
  ) +
  scale_y_discrete(name = "Number of genres per category") +
  scale_x_continuous(
    label = label_number(
      suffix = "M",
      scale = 1e-6
    ),
    name = "Number of ratings",
    breaks = breaks_pretty(n = 5),
    sec.axis = sec_axis(
      trans = ~ . / 600000,
      name = "Average rating score"
    )
  ) +
  theme_1() +
  theme(
    axis.title.x = element_text(color = "#205493"),
    axis.title.x.top = element_text(color = "#651D32"),
    axis.text.x = element_text(color = "#205493"),
    axis.text.x.top = element_text(color = "#651D32")
  )

# Plot graphs together
grid.arrange(gc_movies, gc_ratings,
  ncol = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- Individual-genres, fig.cap = "Distributions of number of movies (A) and number of ratings and average rating scores (B) per individual genre", fig.height = 3.5----

# C. Explore individual genres - distinct genres from original genre categories

# Create modified edx data set with individual genres extracted into individual rows
edx_genres <- edx %>%
  separate_rows(genres,
    sep = "\\|"
  ) %>%
  filter(!genres == "(no genres listed)") # Remove  'no genres listed' category

# 1. Summary

# Summarise number of ratings per individual genre (Action. Drama, etc)
gi_avg_rating <- edx_genres %>%
  group_by(genres) %>%
  summarise(n = n()) %>%
  summarise(
    mean = mean(n),
    sd = sd(n)
  )

# Summarise average rating score per individual genre (Action. Drama, etc)
gi_avg_score <- edx_genres %>%
  group_by(genres) %>%
  summarise(avg_score = mean(rating)) %>%
  summarise(
    mean = mean(avg_score),
    sd = sd(avg_score)
  )

# 2. Compute coefficient of variation (CV)

# Number of ratings per individual genre (Action. Drama, etc)
cv_gi_rating <- gi_avg_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(3)

# Average rating score per individual genre (Action. Drama, etc)
cv_gi_score <- gi_avg_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(4)

# 3. Distributions for number of ratings and average rating score

# Plot distribution of movies across individual genres
gi_movies <- edx_genres %>%
  group_by(genres) %>%
  distinct(movieId, .keep_all = TRUE) %>%
  summarise(n = n()) %>%
  ggplot(aes(
    n,
    reorder(genres, n,
      decreasing = FALSE
    )
  )) +
  geom_col(
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_text(aes(
    label = format(n, big.mark = ","),
    hjust = -0.25
  ),
  size = 2.5
  ) +
  labs(
    title = "A",
    x = "Number of movies",
    y = "Individual genres"
  ) +
  scale_x_continuous(
    label = scales::comma,
    limits = c(0, 6000),
    breaks = breaks_pretty(n = 6)
  ) +
  scale_y_discrete() +
  theme_1()

# Plot distribution of number of ratings and average rating scores across individual genres
gi_ratings <- edx_genres %>%
  group_by(genres) %>%
  summarise(
    n = n(),
    avg_score = mean(rating)
  ) %>%
  ggplot(aes(
    n,
    reorder(genres, n,
      decreasing = FALSE
    )
  )) +
  geom_col(
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_point(aes(x = avg_score * 800000),
    color = "#651D32"
  ) +
  labs(
    title = "B"
  ) +
  scale_y_discrete(name = "Individual genre") +
  scale_x_continuous(
    label = label_number(
      suffix = "M",
      scale = 1e-6
    ),
    name = "Number of ratings",
    breaks = breaks_pretty(n = 5),
    sec.axis = sec_axis(
      trans = ~ . / 600000,
      name = "Average rating score"
    )
  ) +
  theme_1() +
  theme(
    axis.title.x = element_text(color = "#205493"),
    axis.title.x.top = element_text(color = "#651D32"),
    axis.text.x = element_text(color = "#205493"),
    axis.text.x.top = element_text(color = "#651D32")
  )

# Plot graphs together
grid.arrange(gi_movies, gi_ratings,
  ncol = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- Title column---------------------------------------------------------------------------------------------------------------------

# Explore *title* column
###################################################################################

# A. Create release_year column

# Extract year of release from *title* column and save to release_year column
edx <- edx %>%
  extract(title,
    c("title", "release_year"),
    regex = "^(.*)\\s*\\(([0-9]*)\\)\\s*$"
  ) %>%
  mutate(release_year = as.numeric(release_year))

# B. Summary of release_year column:

# 1. Number of ratings
summ_release_year_rating <- edx %>%
  group_by(release_year) %>%
  summarise(
    n = n()
  ) %>%
  summarize(
    mean = mean(n),
    sd = sd(n),
    min = min(n),
    max = max(n)
  )

# 2. Average rating scores
summ_release_year_score <- edx %>%
  group_by(release_year) %>%
  summarise(
    avg_score = mean(rating)
  ) %>%
  summarize(
    mean = mean(avg_score),
    sd = sd(avg_score),
    min = min(avg_score),
    max = max(avg_score)
  )

# C. Compute coefficient of variation (CV) for release year

# 1. Number of ratings per year
cv_release_year_rating <- summ_release_year_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# 2. Average rating score per year
cv_release_year_score <- summ_release_year_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)


## ---- Release-year, fig.cap = "Distribution of number of ratings and average rating scores by movie release year", fig.height = 3------

# D. Distribution across release_year column:

# Plot number of ratings and average rating scores across release years
edx %>%
  group_by(release_year) %>%
  summarise(
    n = n(),
    avg_score = mean(rating)
  ) %>%
  ggplot(aes(x = release_year)) +
  geom_col(aes(y = n),
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_point(aes(y = avg_score * 160000),
    color = "#651D32",
    alpha = 0.5
  ) +
  # Add a line to show mean yearly number of ratings
  geom_hline(
    yintercept = summ_release_year_rating$mean,
    linetype = "dashed",
    color = "#205493"
  ) +
  geom_label(aes(
    1920,
    110000
  ),
  label = paste0(
    "Mean yearly number \n of ratings: ",
    format(round(summ_release_year_rating$mean, 0),
      big.mark = ","
    )
  ),
  color = "#205493",
  fill = "white",
  size = 2
  ) +
  # Add a line to show average yearly rating score
  geom_hline(
    yintercept = summ_release_year_score$mean * 160000,
    linetype = "dashed",
    color = "#651D32"
  ) +
  geom_label(aes(
    2005,
    640000
  ),
  label = paste0(
    "Mean yearly \n rating score: ",
    round(summ_release_year_score$mean, 2)
  ),
  color = "#651D32",
  fill = "white",
  size = 2
  ) +
  labs(
    caption = "Source: MovieLens 10M data set",
    x = "Year of release",
    y = "Number of ratings"
  ) +
  scale_y_continuous(
    label = label_number(
      suffix = "k",
      scale = 1e-3
    ),
    name = "Number of ratings",
    sec.axis = sec_axis(
      trans = ~ . / 160000,
      name = "Average rating score"
    )
  ) +
  scale_x_continuous(
    guide = guide_axis(angle = 90),
    breaks = breaks_pretty(n = 31)
  ) +
  theme_1() +
  theme(
    axis.title.y.left = element_text(color = "#205493"),
    axis.title.y.right = element_text(color = "#651D32"),
    axis.text.y.left = element_text(color = "#205493"),
    axis.text.y.right = element_text(color = "#651D32")
  )


## ---- timestamp, fig.cap = "Distribution of number of ratings and average rating scores per rating year", fig.height = 3---------------

# Explore *timestamp* column
###################################################################################

# A. Create rating_year column:

# Timestamp column units are seconds. Convert column to date & time format (with year as unit) and save this to new column *rating_date*.
edx <- edx %>%
  mutate(rating_year = round_date(as_datetime(timestamp),
    unit = "year"
  )) %>%
  select(-timestamp) # remove timestamp column

# B. Summary of rating_year column:

# Number of ratings
summ_rating_year_rating <- edx %>%
  group_by(rating_year) %>%
  summarise(
    n = n()
  ) %>%
  summarize(
    mean = mean(n),
    sd = sd(n),
    min = min(n),
    max = max(n)
  )

# Average rating scores
summ_rating_year_score <- edx %>%
  group_by(rating_year) %>%
  summarise(
    avg_score = mean(rating)
  ) %>%
  summarize(
    mean = mean(avg_score),
    sd = sd(avg_score),
    min = min(avg_score),
    max = max(avg_score)
  )

# C. Compute coefficient of variation (CV) for rating_year

# 1. Number of ratings per year
cv_rating_year_rating <- summ_rating_year_rating %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# 2. Average rating score per year
cv_rating_year_score <- summ_rating_year_score %>%
  summarise(cv = sd / mean) %>%
  pull(cv) %>%
  round(2)

# D. Distribution across new *date* column:

# Plot number of ratings and average rating scores across rating years
edx %>%
  group_by(rating_year) %>%
  summarise(
    n = n(),
    avg_score = mean(rating)
  ) %>%
  ggplot(aes(x = rating_year)) +
  geom_col(aes(y = n),
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  geom_point(aes(y = avg_score * 240000),
    color = "#651D32"
  ) +
  geom_smooth(aes(y = avg_score * 240000),
    color = "#651D32"
  ) +
  labs(
    caption = "Source: MovieLens 10M data set",
    x = "Year of rating",
    y = "Number of ratings"
  ) +
  scale_y_continuous(
    label = scales::comma,
    name = "Number of ratings",
    sec.axis = sec_axis(
      trans = ~ . / 240000,
      name = "Average rating score"
    )
  ) +
  scale_x_datetime(
    guide = guide_axis(angle = 90),
    breaks = breaks_pretty(n = 14)
  ) +
  theme_1() +
  theme(
    axis.title.y.left = element_text(color = "#205493"),
    axis.title.y.right = element_text(color = "#651D32"),
    axis.text.y.left = element_text(color = "#205493"),
    axis.text.y.right = element_text(color = "#651D32")
  )


## ---- CV-table-------------------------------------------------------------------------------------------------------------------------

# Create table showing CV for number of ratings and average ratings scores for all variables

tibble(
  Variable = c(
    "Movie",
    "User",
    "Genre - original genre categories",
    "Genre - number-of-genres categories",
    "Genre - Individual genres",
    "Release year",
    "Rating date"
  ),
  CV_ratings = round(
    c(
      cv_movie_rating,
      cv_user_rating,
      cv_g_rating,
      cv_gc_rating,
      cv_gi_rating,
      cv_release_year_rating,
      cv_rating_year_rating
    ),
    4
  ),
  CV_avg_rating_score = round(c(
    cv_movie_score,
    cv_user_score,
    cv_g_score,
    cv_gc_score,
    cv_gi_score,
    cv_release_year_score,
    cv_rating_year_score
  ), 4)
) %>%
  kbl(
    caption = "Comparison of CV for number of ratings and average rating scores for all independent variables",
    col.names = c("Variable", "CV (Number of ratings)", "CV (Average rating score)"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped"),
    font_size = 8
  ) %>%
  row_spec(1:3, bold = TRUE)


## ---- Clear memory---------------------------------------------------------------------------------------------------------------------

# Remove temp files and plots to conserve memory
rm(nonzero_entries, tot_entries, sparsity, users, summ_rating, summ_movie_rating, summ_movie_score, cv_movie_rating, cv_movie_score, ratings_m, scores_m, top_5, bottom_5, summ_user_rating, summ_user_score, cv_user_rating, cv_user_score, ratings_u, scores_u, edx_genres, edx_n_genres, g_avg_rating, g_avg_score, cv_g_rating, cv_g_score, ratings_g, scores_g, gc_avg_rating, gc_avg_score, cv_gc_rating, cv_gc_score, gc_movies, gc_ratings, gi_avg_rating, gi_avg_score, cv_gi_rating, cv_gi_score, gi_movies, gi_ratings, summ_release_year_rating, summ_release_year_score, cv_release_year_rating, cv_release_year_score, summ_rating_year_rating, summ_rating_year_score, cv_rating_year_rating, cv_rating_year_score)


## ---- modify edx-----------------------------------------------------------------------------------------------------------------------

# Modify edx data set for modeling - Keep only the columns needed for the modeling approach section
edx <- edx %>%
  select(movieId, userId, title, genres, rating) %>%
  # Remove observations with 'no genre listed' category
  filter(!genres == "(no genres listed)")


## ---- Train-test-sets------------------------------------------------------------------------------------------------------------------

# Split edx data set into train set (80%) and test set (20%)
set.seed(2000, sample.kind = "Rounding") # Set.seed for reproducibility
train_index <- createDataPartition(edx$rating, times = 1, p = 0.8, list = FALSE)
train_set <- edx[train_index, ]
test_set <- edx[-train_index, ]

# Make sure userId and movieId included in the test set is also in the train set
test_set_temp <- test_set %>%
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set and rename
removed <- anti_join(test_set, test_set_temp)
test_set <- test_set_temp
train_set <- rbind(train_set, removed)

# Remove all unnecessary files
rm(test_set_temp, removed, train_index)

# Tabulate dimensions of train and test set
data.frame(
  row.names = c("Train set", "Test set"),
  r_num = c(dim(train_set)[1], dim(test_set)[1]),
  prop = c(
    (nrow(train_set) / sum(nrow(train_set), nrow(test_set))),
    (nrow(test_set) / sum(nrow(train_set), nrow(test_set)))
  )
) %>%
  mutate(
    r_num = format(r_num, big.mark = ","),
    prop = paste(round(prop * 100, digits = 1), "%")
  ) %>%
  kbl(
    caption = "Dimensions and proportions of train and test sets",
    col.names = c("Number of rows", "Proportion of edx"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped"),
    font_size = 8
  )


## ---- RMSE-----------------------------------------------------------------------------------------------------------------------------

# Function to compute RMSE for all models
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## ---- Baseline model-------------------------------------------------------------------------------------------------------------------

# Compute average rating (mu_hat) for all movies across all users
mu_hat <- mean(train_set$rating)

# Compute RMSE for baseline model
avg_model_rmse <- RMSE(test_set$rating, mu_hat)


## ---- Movie effect estimates-----------------------------------------------------------------------------------------------------------

# Compute movie effect estimates
movie_effects <- train_set %>%
  group_by(movieId) %>%
  summarise(mov = mean(rating - mu_hat))

# Plot movie effect estimates
plot_m <- movie_effects %>%
  ggplot(aes(mov)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  # Add line showing median rating
  geom_vline(
    xintercept = median(movie_effects$mov),
    linetype = "dashed",
    color = "black",
    show.legend = TRUE
  ) +
  geom_label(aes(-1, 810),
    label = paste0("Median = ", round(median(movie_effects$mov), 3)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  annotate("text",
    x = -2.5,
    y = 600,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", round(min(movie_effects$mov), 2),
      "\n Mean = ", round(mean(movie_effects$mov), 3),
      "\n Std. Dev. = ", round(sd(movie_effects$mov), 3),
      "\n Maximum = ", round(max(movie_effects$mov), 2)
    )
  ) +
  labs(
    x = "Movie effect estimates",
    y = "Number of movies"
  ) +
  scale_y_continuous(limits = c(0, 820)) +
  theme_1()


## ---- Baseline + Movie effects model - predicted ratings and RMSE----------------------------------------------------------------------

# Predict ratings in test set using Baseline + Movie effects model
predicted_ratings <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  mutate(prediction = mu_hat + mov) %>%
  pull(prediction)

# Compute RMSE for Baseline + Movie effects model
movie_model_rmse <- RMSE(
  test_set$rating,
  predicted_ratings
)


## ---- User effect estimates------------------------------------------------------------------------------------------------------------

# Compute user effect estimates
user_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarise(use = mean(rating - mu_hat - mov))

# Plot user effect estimates
plot_u <- user_effects %>%
  ggplot(aes(use)) +
  geom_histogram(
    binwidth = 0.1,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  # Add line showing mean rating
  geom_vline(
    xintercept = mean(user_effects$use),
    linetype = "dashed",
    color = "black",
    show.legend = TRUE
  ) +
  geom_label(aes(1, 7500),
    label = paste0("Median = ", round(median(user_effects$use), 4)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  annotate("text",
    x = -2.5,
    y = 4000,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", round(min(user_effects$use), 2),
      "\n Mean = ", round(mean(user_effects$use), 4),
      "\n Std. Dev. =", round(sd(user_effects$use), 3),
      "\n Maximum = ", round(max(user_effects$use), 2)
    )
  ) +
  labs(
    x = "User effect estimates",
    y = "Number of users"
  ) +
  scale_y_continuous(limits = c(0, 8000)) +
  theme_1()


## ---- Baseline + Movie + User effects model - predicted ratings and RMSE---------------------------------------------------------------

# Predict ratings in test set using Baseline + Movie + User effects model
predicted_ratings <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  mutate(prediction = mu_hat + mov + use) %>%
  pull(prediction)

# Compute RMSE for Baseline + Movie + User effects model
user_model_rmse <- RMSE(
  test_set$rating,
  predicted_ratings
)


## ---- Genre effect estimates (Original genre categories)-------------------------------------------------------------------------------

# Compute genre effect estimates - Original genre categories
genre_effects <- train_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  group_by(genres) %>%
  summarize(g_effects = mean(rating - mu_hat - mov - use))

# Graph genre effect estimates (Original genre categories)
plot_g <- genre_effects %>%
  ggplot(aes(g_effects)) +
  geom_histogram(
    binwidth = 0.03,
    fill = "#205493",
    color = "white",
    alpha = 0.5
  ) +
  # Add line showing mean rating
  geom_vline(
    xintercept = mean(genre_effects$g_effects),
    linetype = "dashed",
    color = "black",
    show.legend = TRUE
  ) +
  geom_label(aes(0.15, 2250),
    label = paste0("Median = ", round(median(genre_effects$g_effects), 4)),
    color = "grey50",
    fill = "white",
    size = 2
  ) +
  annotate("text",
    x = 0.5,
    y = 150,
    size = 2,
    color = "grey50",
    label = str_c(
      "Minimum = ", round(min(genre_effects$g_effects), 3),
      "\n Mean = ", round(mean(genre_effects$g_effects), 4),
      "\n Std. Dev. = ", round(sd(genre_effects$g_effects), 4),
      "\n Maximum = ", round(max(genre_effects$g_effects), 3)
    )
  ) +
  labs(
    x = "Genre effect estimates",
    y = "Number of original genre categories"
  ) +
  scale_y_continuous(limits = c(0, 250)) +
  theme_1()


## ---- Baseline + Movie + User + Genre effects model (Original genre categories) - predicted ratings and RMSE---------------------------

# Predict ratings in test set using Baseline + Movie + User + Genre effects model (Original genre categories)
predicted_ratings <- test_set %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  left_join(genre_effects, by = "genres") %>%
  mutate(prediction = mu_hat + mov + use + g_effects) %>%
  pull(prediction)

# Compute RMSE for Baseline + Movie + User + Genre effects model (Original genre categories)
genre_model_rmse1 <- RMSE(
  test_set$rating,
  predicted_ratings
)


## ---- Optimal-tuning-parameter, fig.cap = "Optimal tuning parameter (lambda)", fig.height = 3.5----------------------------------------

# Generate sequence of lambdas within which optimal lambda lies
lambdas <- seq(0, 10, 0.1)

# Create function to apply sequence of lambdas to effect estimates and compute resulting RMSE for each lambda in the sequence
rmses <- sapply(lambdas, function(l) {

  # Compute movie effect estimates for the train set
  movie_effects <- train_set %>%
    group_by(movieId) %>%
    summarise(mov = sum(rating - mu_hat) / (n() + l))
  # Compute user effect estimates for the train set
  user_effects <- train_set %>%
    left_join(movie_effects, by = "movieId") %>%
    group_by(userId) %>%
    summarise(use = sum(rating - mu_hat - mov) / (n() + l))
  # Compute genre effect estimates for the train set (using  Baseline + Movie + User + Genre effects model (Original genre categories))
  genre_effects <- train_set %>%
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    group_by(genres) %>%
    summarise(g_effects = sum(rating - mu_hat - mov - use) / (n() + l))

  # Predict ratings for the test set using the above movie + user + genre effect estimates and return RMSEs
  predicted_ratings <- test_set %>%
    left_join(movie_effects, by = "movieId") %>%
    left_join(user_effects, by = "userId") %>%
    left_join(genre_effects, by = "genres") %>%
    mutate(prediction = mu_hat + mov + use + g_effects) %>%
    pull(prediction)

  return(RMSE(predicted_ratings, test_set$rating))
})

# Extract optimal tuning parameter (lambda) - i.e., lambda which gives the minimum RMSE
lambda <- lambdas[which.min(rmses)]

# Extract RMSE for Regularised Baseline + Movie + User + Genre effects model (Original genre categories)
reg_model_rmse <- min(rmses)

# Plot sequence of lambdas and corresponding RMSE
data.frame(
  lambdas = lambdas,
  rmses = rmses
) %>%
  ggplot(aes(lambdas, rmses)) +
  geom_point(color = "#205493") +
  # Add line showing optimal lambda
  geom_vline(
    xintercept = lambda,
    linetype = 2
  ) +
  # Label line showing optimal lambda
  geom_label(
    label = paste0(
      "optimal lambda = ",
      lambda
    ),
    x = 5.5,
    y = 0.8649,
    size = 2,
    color = "grey50"
  ) +
  # Add line showing RMSE associated with optimal lambda
  geom_hline(
    yintercept = min(rmses),
    linetype = 2
  ) +
  # Label line showing RMSE associated with optimal lambda
  geom_label(
    label = paste0(
      "RMSE for optimal \n lambda = ",
      round(reg_model_rmse, 5)
    ),
    x = 8.5,
    y = 0.8645,
    size = 2,
    color = "grey50"
  ) +
  labs(
    caption = "Source: MovieLens 10M data set",
    x = "Tuning parameter (lambda)",
    y = "RMSE (Cross-validation)"
  ) +
  theme_1()


## ---- Regularisation, fig.cap = "Comparison between regularised and non-regularised movie effect estimates by sample sizes (number of ratings/users)", fig.height = 3.5----

# Show how small samples result in large estimates and how regularisation shrinks the estimates closer to zero.

# # For train set, compute movie effect estimates before and after regularisation and compare.

# Compute movie effect estimates before regularisation
mov_ori <- train_set %>%
  group_by(movieId) %>%
  summarise(ori_effects = mean(rating - mu_hat))

# Compute movie effect estimates after regularisation
mov_reg <- train_set %>%
  group_by(movieId) %>%
  summarize(reg_effects = sum(rating - mu_hat) / (n() + lambda), n_i = n())

# Plots regularised vs non-regularised movie effect estimates for train set
tibble(
  before = mov_ori$ori_effects,
  after = mov_reg$reg_effects,
  n = mov_reg$n_i
) %>%
  ggplot(aes(
    x = before,
    y = after,
    size = n
  )) +
  geom_point(
    color = "#205493",
    shape = 1,
    alpha = 0.5
  ) +
  geom_abline(aes(
    intercept = 0,
    slope = 1
  ),
  color = "#651D32"
  ) +
  geom_hline(
    yintercept = 0,
    linetype = 2
  ) +
  labs(
    caption = "Source: MovieLens 10M data set",
    x = "Non-regularised estimates",
    y = "Regularised estimates",
    size = "Number of ratings/users"
  ) +
  scale_x_continuous(
    limits = c(-3.5, 2.5),
    breaks = breaks_pretty(n = 5)
  ) +
  scale_y_continuous(
    limits = c(-3.5, 2.5),
    breaks = breaks_pretty(n = 5)
  ) +
  theme_1()


## ---- Modify validation data set-------------------------------------------------------------------------------------------------------

# Modify validation data set for evaluation

# Keep only the columns needed for the modeling approach section
validation <- validation %>%
  # Extract release year into its own column and leave only movie title in title column
  extract(title,
    c("title", "release_year"),
    regex = "^(.*)\\s*\\(([0-9]*)\\)\\s*$"
  ) %>%
  mutate(release_year = as.numeric(release_year)) %>%
  # Keep only the same columns as edx
  select(movieId, userId, title, genres, rating) %>%
  # Remove observations with 'no genre listed' category
  filter(!genres == "(no genres listed)")


## ---- Movie-effects-examples-----------------------------------------------------------------------------------------------------------

# Compute movie effect estimates for the movies Titanic and Tokyo!
movie_effects <- edx %>%
  # select only Titanic and Tokyo! movieIds
  filter(movieId %in% c(1721, 63179)) %>%
  group_by(movieId) %>%
  summarise(movie_effects = mean(rating - mu_hat))


## ---- User-effects-examples------------------------------------------------------------------------------------------------------------

# Compute movie effect estimates for the movies rated by 2 users
movie_effects <- edx %>%
  # select only movies rated bu the users 59269 & 22170
  filter(userId %in% c(59269, 22170)) %>%
  group_by(movieId) %>%
  summarise(movie_effects = mean(rating - mu_hat))

# Compute user effect estimates for the 2 users
user_effects <- edx %>%
  filter(userId %in% c(59269, 22170)) %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarise(user_effects = mean(rating - mu_hat - movie_effects))


## ---- Genre-effects-examples-----------------------------------------------------------------------------------------------------------

# Compute movie effect estimates for all movies in the 2-genre category
movie_effects <- edx %>%
  filter(genres %in% c("Drama", "Action|Animation|Comedy|Horror")) %>%
  group_by(movieId) %>%
  summarise(movie_effects = mean(rating - mu_hat))

# Compute user effect estimates all users who rated movies in the 2-genre category
user_effects <- edx %>%
  filter(genres %in% c("Drama", "Action|Animation|Comedy|Horror")) %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarise(user_effects = mean(rating - mu_hat - movie_effects))

# Compute genre effect estimates for the 2-genre category
genre_effects <- edx %>%
  filter(genres %in% c("Drama", "Action|Animation|Comedy|Horror")) %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  group_by(genres) %>%
  summarise(gen_effects = mean(rating - mu_hat - movie_effects - user_effects))


## ---- Plot-effects, fig.cap = "Distribution of effect estimates for movies, users and original genre categories", fig.height = 4-------

# Plot all effect estimates (movie, user and genre) in one layout
grid.arrange(plot_m, plot_u, plot_g,
  ncol = 2,
  nrow = 2,
  bottom = textGrob("Source: MovieLens 10M data set",
    hjust = 1,
    x = 1,
    gp = gpar(fontsize = 6)
  )
)


## ---- RMSE-results---------------------------------------------------------------------------------------------------------------------

# Create table to compile RMSE results for all models
# Add project target RMSE for comparison
rmse_table <- tibble(
  method =
    c(
      "Target",
      "Baseline model",
      "Baseline + Movie Effect Model",
      "Baseline + Movie + User Effect Model",
      "Baseline + Movie + User + Genre Effect Model",
      "Regularised Baseline + Movie + User + Genre Effect Model"
    ),
  RMSE = round(
    c(
      0.86490,
      avg_model_rmse,
      movie_model_rmse,
      user_model_rmse,
      genre_model_rmse1,
      reg_model_rmse
    ), 5
  )
)

rmse_table %>%
  kbl(
    caption = "Comparison of RMSE results from different modeling approaches",
    col.names = c("Modeling Approach", "RMSE"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position", "striped"),
    font_size = 8
  ) %>%
  row_spec(c(1, 6), bold = TRUE)


## ---- Validation-----------------------------------------------------------------------------------------------------------------------

# To evaluate the final algorithm:
# 1. Use the regularised movie + user + genre effects model (Original genre categories) and optimal lambda to estimate movie, user and genre effects in the full edx data set,
# 2. Predict ratings for the validation set and,
# 3. Compute RMSE for the final model, comparing predicted ratings to the ratings in the validation set .

# Compute movie, user and genre effect estimates using optimal lambda and edx data set
mu_hat <- mean(edx$rating)

movie_effects <- edx %>%
  group_by(movieId) %>%
  summarise(mov = sum(rating - mu_hat) / (n() + lambda))

user_effects <- edx %>%
  left_join(movie_effects, by = "movieId") %>%
  group_by(userId) %>%
  summarise(use = sum(rating - mu_hat - mov) / (n() + lambda))

genre_effects <- edx %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  group_by(genres) %>%
  summarize(g_effects = sum(rating - mu_hat - mov - use) / (n() + lambda))

# Predict ratings for validation data set
predicted_ratings <- validation %>%
  left_join(movie_effects, by = "movieId") %>%
  left_join(user_effects, by = "userId") %>%
  left_join(genre_effects, by = "genres") %>%
  mutate(prediction = mu_hat + mov + use + g_effects) %>%
  pull(prediction)

# Compute final model RMSE
final_rmse <- RMSE(predicted_ratings, validation$rating)

# Create rmse results table comparing final model to project target
tibble(
  method =
    c(
      "Target",
      "Final (Validation) Model"
    ),
  RMSE = round(
    c(
      0.86490,
      final_rmse
    ), 5
  ),
  diff = c(0, paste0(round(((final_rmse - 0.86490) / 0.86490) * 100, 3), "%"))
) %>%
  kbl(
    caption = "Comparison of RMSEs from final model RMSE to project target",
    col.names = c("Modeling Approach", "RMSE", "Difference"),
    booktabs = TRUE
  ) %>%
  kable_classic(
    latex_options = c("hold_position"),
    font_size = 8
  ) %>%
  row_spec(2, color = "white", background = "black")


