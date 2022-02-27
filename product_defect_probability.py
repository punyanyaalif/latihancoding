import scipy.stats as stats
import numpy as np

### Task Group 1 ###
## rate parameter of distribution 
lam = 13
## probability of observing exactly 13 defects on given day
x = stats.poisson.pmf(lam, lam)
print(x)
## how often that having 4 or fewer defects a day
y = stats.poisson.cdf(4, lam)
print(y)
## how often that having 9 or more defects a day
z = 1 - stats.poisson.cdf(9, lam)
print(z)

### Task Group 2 ###
## 365 random values 
year_defects = stats.poisson.rvs(lam, size = 365)
## print first 20 values in data set
print(year_defects[0:20])
## total number of defects we would expect over 365 days
b = (lam*365)
print(b)
## total sum of the data set year_defects
c = sum(year_defects)
print(c)
## average number of defects per day
d = np.mean(year_defects)
print(d)
## highest amount of defect 
e = year_defects.max()
print(e)
## the probability of observing that maximum value or more from the Poisson(13) distribution
f = 1 - (stats.poisson.cdf(year_defects.max(), lam))
print(f)

### Extra Bonus ###
# the number of defects that would put us in the 90th percentile for a given day
print(stats.poisson.ppf(0.9, lam))

# what proportion of our simulated dataset year_defects is greater than or equal to the number we calculated in the previous step
print(sum(year_defects > stats.poisson.ppf(0.9, lam))/len(year_defects))