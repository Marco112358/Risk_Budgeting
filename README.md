# Risk_Budgeting
 
 This report is a mess. 
The main function runs through a risk balancing model (u choose the risk weights not capital weights.
It then runs through time using exponential weighting to get the backward looking covariance matrix.
Then it rebalances (currently every 30 days) to get the new capital weights for the risk weighting.
There is also the "rebal" file that tests different rebalancing strategies. Some based on days, some based on tolerance levels (absolute and relative).
The rest is necessary functions.

 Risk balancing uses a version of Newtons Method to get the weights within a tolerance
