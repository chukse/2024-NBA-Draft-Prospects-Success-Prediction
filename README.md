# 2024 NBA Draft Prospects Success Prediction

This project aims to predict the success of NBA draft prospects using historical data and machine learning models. The main script processes data, trains a model, and predicts the success of current draft prospects.

## Table of Contents

- [Project Description](#project-description)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

This project uses historical NBA data and machine learning to predict various performance metrics for NBA draft prospects. The script performs the following tasks:

1. Load and clean datasets.
2. Normalize player names.
3. Merge datasets using fuzzy matching.
4. Train a machine learning model.
5. Predict the success of current draft prospects.
6. Sort and export the results.

## Data

The following datasets are used in this project:

1. `draft-data-20-years.csv`
2. `nba_draft_prospects.csv`
3. `2024_NBA_Draft_results.csv`
4. `CollegeBasketballPlayers2009-2021_full.csv`
5. `av_modern_RAPTOR_by_player.csv`

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/nba-draft-prospects.git
cd nba-draft-prospects
pip install -r requirements.txt
 
