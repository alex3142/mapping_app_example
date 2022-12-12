# Data and Constants Explanation

This file contains the description of the data and constants in the 
fever data science coding challenge.

## constants.yaml

This file contains constants in a key value form where the key provides the
name of the constant and its units. Rationale behind the constants can be found below.

long scale
`distance((51.451, -0.236313), (51.451, 0.005536)).meters = 16812.114800647625`

lat_scale
`distance((51.451, -0.236313), (51.560, -0.236313)).meters = 12127.115295910631`

#### satellite_data
This is the inputs to the normal distribution describing the satellite path
The mean is at zero distance from the path and 
as it is known that 95% of the probability is within `+/- 3160m `of the satellite path
(orientated arbitrarily above and below path)
this means that 1.96 standard deviation (sigma) therefore `sigma = 3160/1.96 = 1612.2448979591836`

#### river_data
Similar rationale as described in the above satellite data the mean here is zero and the standard deviation is 
 `sigma = 2730/1.96 = 1392.857142857143`

#### b_o_e_data
This is the bank of england distribution inputs.
Given this distribution is lognormal with a mean of `4744` and a mode of `3777`.
and the mean of a lognormal distribution is `exp(mu + (sigma^2 / 2)`, and the 
mode is `exp(mu - sigma^2)`.
Setting `4744 = exp(mu + (sigma^2 / 2)` and `3777 = exp(mu - sigma^2)` 
and performing algebraic manipulation it can be shown that
`mu = 8.38865240135586` and `sigma = 0.3898295507569983`

## lat_long_coordinates_research_location.csv

This file contains the lat long data provided in the project document 
however in a csv form to be more easily used, it contains the following columns:
- lat: latitude (assumed to be WGS84)
- long - longitude (assumed to be WGS84)
- description - information on what the lat longs on each row represent containing the following values:
  - bank_of_england - position of Bank of England
  - satellite_path - points on a great circle representing the satellite path
  - river_thames - set of coordinates between which the river Thames is assumed to be piecewise linear
