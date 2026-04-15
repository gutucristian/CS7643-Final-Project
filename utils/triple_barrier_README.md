# Triple Barrier Labeling

The triple_barrier.py script generates buy, sell, and hold labels using the triple barrier method.

## Methodology

For each day, we treat the current price as the entry point and look ahead a fixed number of days.

The following inputs define the labeling:
- **upper barrier** - profit target (percentage above entry price)
- **lower barrier** - stop loss (percentage below entry price)
- **time limit** - max holding period (number of days to look ahead)

We then check each future day in order until a barrier is hit or the time limit is reached.
- if price hits the upper barrier first → label = 1
- if price hits the lower barrier first → label = -1
- if neither is hit within the time window → label = 0

High prices are used to check the upper barrier, and low prices are used for the lower barrier (i.e., whether the price ever crossed the threshold during the day).

## Label Meaning

- **1 (Buy)** → price moved up enough to hit the profit target  
- **-1 (Sell)** → price dropped enough to hit the stop loss  
- **0 (Hold)** → no strong movement within the time window  

## Current Labeling Hyperparameters

- upper barrier: +3%  
- lower barrier: -3%  
- holding period: 10 days  

This means each label answers "will the price move ±3% within the next 10 days?"