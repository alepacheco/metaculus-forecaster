# Releases

## v1

First version. Forecasts every question type the tournaments use (binary, multiple choice,
numeric, discrete, date) from two research sources (a news search and a web-search brief) and
an ensemble of five forecasts from models of several labs, combined by median (binary,
multiple choice) or by averaging distributions (numeric, discrete, date), with probabilities
kept between 2% and 98% and a 1% floor on each option. Posts one forecast and one private
comment per question, finishes an interrupted question from its own record, and keeps to the
season's budget: a cheaper ensemble near the limit, no more spending at it. With no budget
for the season it forecasts with no-cost models only (three labs, news search for research).
