#!/usr/bin/env Rscript

# Build person-level smoking history text from UKB extracted phenotype file.
# Output: data frame with two columns: IID, Text

args <- commandArgs(trailingOnly = TRUE)

default_input <- "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/extracted_pheno_smoke.txt"
default_output <- "/proj/yunligrp/users/qinwen/R_works/LLM_embedding/data/ukb_phenotype/smoke_text_df.tsv"

input_path <- if (length(args) >= 1) args[[1]] else default_input
output_path <- if (length(args) >= 2) args[[2]] else default_output

message("Reading: ", input_path)
raw_df <- read.delim(
  input_path,
  header = TRUE,
  sep = "\t",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

if (ncol(raw_df) < 16) {
  stop("Input file must have at least 16 columns (IID + 15 smoking phenotypes).")
}

# Keep original order from extraction script:
# IID,1239,1249,1259,1269,1279,2644,2867,2877,2897,2907,3436,3446,3456,3476,3486
names(raw_df)[1:16] <- c(
  "IID", "current_tobacco", "past_tobacco", "smokers_household",
  "exposure_home", "exposure_outside", "light_smoker_100",
  "age_started_former", "type_tobacco_former", "age_stopped",
  "stopped_6m", "age_started_current", "type_tobacco_current",
  "cigs_daily_current", "difficulty_1day", "tried_stop"
)

as_clean_num <- function(x) {
  # UKB negative values are usually special-missing codes; treat as unknown.
  y <- suppressWarnings(as.numeric(x))
  ifelse(is.na(y) | y < 0, NA_real_, y)
}

as_clean_int <- function(x) {
  y <- as_clean_num(x)
  ifelse(is.na(y), NA_integer_, as.integer(y))
}

describe_binary <- function(x, yes_text, no_text, unknown_text) {
  if (is.na(x)) {
    return(unknown_text)
  }
  if (x == 1) {
    return(yes_text)
  }
  if (x == 0) {
    return(no_text)
  }
  paste0(unknown_text, " (code=", x, ").")
}

describe_current_tobacco <- function(x) {
  if (is.na(x)) return("Current tobacco smoking status is unknown.")
  if (x == 0) return("The person does not currently smoke tobacco.")
  if (x == 1) return("The person currently smokes tobacco on most or all days.")
  if (x == 2) return("The person currently smokes tobacco occasionally.")
  paste0("Current tobacco smoking status is unknown (code=", x, ").")
}

describe_past_tobacco <- function(x) {
  if (is.na(x)) return("Past tobacco smoking history is unknown.")
  if (x == 0) return("No past tobacco smoking is reported.")
  if (x == 1) return("The person smoked tobacco on most or all days in the past.")
  if (x == 2) return("The person smoked tobacco occasionally in the past.")
  if (x == 3) return("The person tried tobacco once or twice in the past.")
  if (x == 4) return("The person reports never smoking tobacco in the past.")
  paste0("Past tobacco smoking history is unknown (code=", x, ").")
}

describe_status <- function(curr, past, age_former, age_current, cigs_daily) {
  has_current_detail <- !is.na(age_current) || !is.na(cigs_daily)
  has_former_detail <- !is.na(age_former)

  is_current <- (!is.na(curr) && curr %in% c(1, 2)) || has_current_detail
  is_former <- (!is.na(past) && past %in% c(1, 2, 3)) || has_former_detail
  is_never <- (!is.na(past) && past == 4) && !is_current && !is_former

  if (is_current) {
    return("Overall smoking status: current smoker.")
  }
  if (is_never) {
    return("Overall smoking status: likely never smoker.")
  }
  if (is_former) {
    return("Overall smoking status: former or ever smoker (not current).")
  }
  "Overall smoking status is unknown."
}

build_text_one <- function(row) {
  curr <- as_clean_int(row[["current_tobacco"]])
  past <- as_clean_int(row[["past_tobacco"]])
  hh <- as_clean_int(row[["smokers_household"]])
  exp_home <- as_clean_int(row[["exposure_home"]])
  exp_out <- as_clean_int(row[["exposure_outside"]])
  light100 <- as_clean_int(row[["light_smoker_100"]])
  age_former <- as_clean_num(row[["age_started_former"]])
  type_former <- as_clean_int(row[["type_tobacco_former"]])
  age_stop <- as_clean_num(row[["age_stopped"]])
  stop6m <- as_clean_int(row[["stopped_6m"]])
  age_current <- as_clean_num(row[["age_started_current"]])
  type_current <- as_clean_int(row[["type_tobacco_current"]])
  cigs_daily <- as_clean_num(row[["cigs_daily_current"]])
  diff_1day <- as_clean_int(row[["difficulty_1day"]])
  tried_stop <- as_clean_int(row[["tried_stop"]])

  text_parts <- c(
    describe_status(curr, past, age_former, age_current, cigs_daily),
    describe_current_tobacco(curr),
    describe_past_tobacco(past),
    describe_binary(
      hh,
      "There are smokers in the household.",
      "There are no smokers in the household.",
      "Household smoker status is unknown."
    ),
    if (is.na(exp_home)) {
      "Exposure to tobacco smoke at home is unknown."
    } else {
      paste0("Exposure to tobacco smoke at home is recorded with level/code ", exp_home, ".")
    },
    if (is.na(exp_out)) {
      "Exposure to tobacco smoke outside home is unknown."
    } else {
      paste0("Exposure to tobacco smoke outside home is recorded with level/code ", exp_out, ".")
    },
    describe_binary(
      light100,
      "The person is a light smoker with at least 100 lifetime smokes.",
      "The person is not a light smoker with at least 100 lifetime smokes.",
      "Light-smoker (100 lifetime smokes) status is unknown."
    ),
    if (is.na(age_former)) {
      "Age started smoking in former-smoker history is unknown."
    } else {
      paste0("In former-smoker history, smoking started at age ", age_former, ".")
    },
    if (is.na(type_former)) {
      "Type of tobacco previously smoked is unknown."
    } else {
      paste0("Type of tobacco previously smoked is coded as ", type_former, ".")
    },
    if (is.na(age_stop)) {
      "Age stopped smoking is unknown."
    } else {
      paste0("Smoking stopped at age ", age_stop, ".")
    },
    describe_binary(
      stop6m,
      "The person has stopped smoking for at least 6 months before.",
      "The person has not stopped smoking for 6+ months before.",
      "History of stopping smoking for 6+ months is unknown."
    ),
    if (is.na(age_current)) {
      "Age started smoking in current-smoker history is unknown."
    } else {
      paste0("In current-smoker history, smoking started at age ", age_current, ".")
    },
    if (is.na(type_current)) {
      "Type of tobacco currently smoked is unknown."
    } else {
      paste0("Type of tobacco currently smoked is coded as ", type_current, ".")
    },
    if (is.na(cigs_daily)) {
      "Number of cigarettes currently smoked daily is unknown."
    } else {
      paste0("Number of cigarettes currently smoked daily is ", cigs_daily, ".")
    },
    if (is.na(diff_1day)) {
      "Difficulty not smoking for one day is unknown."
    } else {
      paste0("Difficulty not smoking for one day is coded as ", diff_1day, ".")
    },
    describe_binary(
      tried_stop,
      "The person has tried to stop smoking.",
      "The person has not tried to stop smoking.",
      "History of trying to stop smoking is unknown."
    )
  )

  paste(text_parts, collapse = " ")
}

message("Generating text for ", nrow(raw_df), " participants...")
out_df <- data.frame(
  IID = raw_df$IID,
  Text = vapply(seq_len(nrow(raw_df)), function(i) build_text_one(raw_df[i, ]), character(1)),
  stringsAsFactors = FALSE
)

dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.table(
  out_df,
  file = output_path,
  sep = "\t",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

message("Done. Wrote: ", output_path)
message("Output columns: ", paste(colnames(out_df), collapse = ", "))
