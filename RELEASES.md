# Releases

## v2

Keeps the ensemble whole when the provider withdraws a model. One of the no-cost models was
withdrawn early in the season, which left two of the five forecasts failing on every question.
Now the bot reads the provider's list of offered models every hour: a forecaster whose model is
no longer listed, is past its listed expiry, or answers "not found" or "forbidden" is replaced in
the same question by the next model of a reserve, else by another member, so every question
keeps five forecasts; the health file names the models out. The no-cost ensemble was re-picked
by how the organisers' single-model bots of these models scored with news in the last full
season (Summer 2026), among the models that answer this key: the one with a scored season three
times, with two newer models beside it. An empty reply is asked for once more before it counts
as a failure, and each failed forecaster is logged with its place in the ensemble and its model.
Comments put a currency symbol before the number ("$2,500"), a one-off pass takes only the
configured tournaments or the testing area, and the health file's last error comes from every
error logged.

## v1

First version. Forecasts every question type the tournaments use (binary, multiple choice,
numeric, discrete, date) from two research sources (a news search and a web-search brief) and
an ensemble of five forecasts from models of several labs, combined by median (binary,
multiple choice) or by averaging distributions (numeric, discrete, date), with probabilities
kept between 2% and 98% and a 1% floor on each option. Posts one forecast and one private
comment per question, finishes an interrupted question from its own record, and keeps to the
season's budget: a cheaper ensemble near the limit, no more spending at it. With no budget
for the season it forecasts with no-cost models only (three labs, news search for research).
Its health file and log carry counts, costs and error types only, with each failed request's
status code, so a blocked source shows without reading any forecast.
