
def getPresentValue(CF, R, T=None, g=0., tax=0.):
    """
    Annuity: a finite stream of CFs of identical magnitude and equal spacing in time
    Perpetuity: an infinite...

    :param CF: Cash flow identically distributed over time
    :param R: interest rate per annum
    :param g: growing rate of annuity
    :param T: Total number of CF
    :return:
    """
    R = R * (1.-tax)

    if T is None:
        PV = CF/(R-g)
    else:
        PV = CF * (1-((1.+R)/(1.+g))**(-T))/(R-g)
    return PV

def getCashFlow(PV, R, g=0):
    CF = (R-g) * PV
    return CF

"""
Problem 1. Amortizing Bond
Assume that a bond makes 30 equal annual payments of \$1,000$1,000 starting one year from today.
(This security is sometimes referred to as an amortizing bond.)
If the discount rate is 3.5\%3.5% per annum, what is the current price of the bond?
(Hint: Recognize that this cash flow stream is an annuity and that the price of an asset is the present value of its future cash flows.)
"""
print getPresentValue(CF=1000, R=0.035, T=30)

"""
Problem 2. Coupon Bond
Assume that a bond makes 10 equal annual payments of \$1,000$1,000 starting one year from today.
The bond will make an additional payment of \$100,000$100,000 at the end of the last year, year 10.
(This security is sometimes referred to as a coupon bond.)
If the discount rate is 3.5\%3.5% per annum, what is the current price of the bond?
(Hint: Recognize that this bond can be viewed as two cash flow streams: (1) a 10-year annuity with annual payments of \$1,000$1,000, and (2) a single cash flow of \$100,000$100,000 arriving 10 years from today. Apply the tools you've learned to value both cash flow streams separately and then add.)
"""
print getPresentValue(CF=1000, R=0.035, T=10) + getPresentValue(CF=100000, R=(1+0.035)**10, T=1)

"""
Problem 3. Paying for School
Your daughter will start college one year from today, at which time the first tuition payment of $58,000 must be made. 
Assuming that tuition does not increase over time and that your daughter remains in school for four years, 
how much money do you need today in your savings account, earning 5\%5% per annum, 
in order to make the tuition payments over the next four years ?
"""
print getPresentValue(CF=58000, R=0.05, T=4)

"""
Problem 4. Perpetuities
Imagine that the government decided to fund its current deficit of \$431$431 billion dollars by issuing a perpetuity offering a 4\%4% annual return. 
How much would the government have to pay bondholders each year in perpetuity? Express your answer in billions of dollars.
(Hint: The \$431$431 billion is just the present value of these cash flows at a discount rate of 4\%4%.)
"""
print getCashFlow(PV=431, R=0.04)

"""
Problem 5. Paying for School with Taxes
Your daughter will start college one year from today, at which time the first tuition payment of \$58,000$58,000 must be made. 
Assume that tuition does not increase over time and that your daughter remains in school for four years. 
How much money do you need today in your savings account, earning 5\%5% per annum, in order to make the tuition payments over 
the next four years, provided that you have to pay 35\%35% per annum in taxes on any earnings (e.g., interest on the savings)?
"""
print getPresentValue(CF=58000, R=0.05, T=4, tax=0.35)

net_operating_profit_after_taxes = 123.50
depreciation = 18.
increase_in_net_working_capital = 239.
capital_expenditures = 27.5
unlevered_free_cash_flow = net_operating_profit_after_taxes + depreciation - capital_expenditures - increase_in_net_working_capital
print unlevered_free_cash_flow

print 5./(3+2*2) * 0.9 *14./24

"""
Consider an assembly line with 20 stations. Each station has a 0.5% probability of making a defect. 
At the end of the line, an inspection step singles out the defective units. 
The inspection step catches 80% of all defects. 
From inspection, units that are deemed to be non-defective are moved to the shipping department.

If a defect is found at inspection, it is sent to the rework department. 
Rework fixes about 95% of the defective units. 
Units are directly shipped from the rework department with no further inspection taking place.

What is the probability that a unit ends up in rework (in decimal form)?
"""
print 0.005 * 0.8

"""
What is the probability that a defective unit is shipped (in decimal form)?
"""
print 0.2 + 0.8*0.05

"""
Consider the following three-step assembly operation with quality problems. 
All resources are staffed by one employee.

The first resource has a processing time of 7 minutes per unit.
The second resource has a processing time of 6 minutes per unit.
The third resource has a processing time of 5 minutes per unit. With a 40% probability, 
the flow unit coming out of the third resource has to be reworked. 
In that case, the operations at the second and third resources are repeated. 
You can assume that (a) rework always succeeds 
(i.e. a unit going through the rework loop will always work after the third resource),and 
(b) the processing times for units in rework are the same as for regular units.
For every unit of demand, how many units have to flow through the second step in the process?
"""


def getCapabilityScore(Lower_Specification_Level, Upper_Specification_Level, standard_deviation):
    return (Upper_Specification_Level-Lower_Specification_Level)/(6.*standard_deviation)

def getSD(Lower_Specification_Level, Upper_Specification_Level, CapabilityScore):
    return (Upper_Specification_Level-Lower_Specification_Level)/(6.*CapabilityScore)

print getCapabilityScore(0.45, 0.55, 0.02)

from scipy.stats import norm
print norm.cdf((0.45-0.5)/0.02) * 2

print getCapabilityScore(0.45, 0.55, 4./3)


"""
Consider an assembly line with 50 steps that are carried out sequentially. 
Each of the 50 steps has a defect probability of 1%. 
The final product produced on the assembly line is defective if any one of the 50 steps made a defect.

At the end of the assembly line there are two operators independently inspecting the product. 
Each of them is recognizing a defective product with a 90% probability. 
Assuming that the product is not defective, it is moved to the shipping department. 
Otherwise, the product is scrapped.

What is the probability that a defective product is produced (independent of the defect being found or not)?
"""
print 1 - 0.99**50

"""
What is the probability that a defective product is moved to the shipping department?
"""
print (1 - 0.99**50) * 0.1**2

"""
Consider the following four-step assembly operation with quality problems:

- The first resource has a processing time of 5 minutes per unit and one employee doing the operation.
- The second resource has a processing time of 4 minutes per unit. 
It also has one employee doing the operation. 
However, this is a very delicate task and 80% of all products have to be scrapped after this step.
- Two workers are staffed for the third resource. 
No quality problems occur at this resource and the processing time is 20 minutes per unit.
- At the fourth and final resource, one operator handles the product. 
No quality problems exist at this step and the processing time is 12 minutes per unit.

For every unit of demand, how many units have to flow through the second step in the process?
"""
success_rate = 0.2
print 1./success_rate

""" Where in the process is the bottleneck? """


"""
The management of a large fast food chain evaluates the supplier of their deep-frozen beef patties. 
The supplier delivers frozen beef patties that according to the specifications have to weigh 
between 94.5 grams and 95.5 grams. All other beef patties are rejected.

A random sample of 30 patties that were inspected over the course of the last week revealed 
that the average weight was 95.0 grams. The standard deviation was 0.25 grams.

What is the capability score of the supplier?
"""
print getCapabilityScore(Lower_Specification_Level=94.5, Upper_Specification_Level=95.5, standard_deviation=0.25)

"""
What percentage of the deliveries is likely to be outside the specification limits 
(outside the interval of [94.5, 95.5])?"""
print norm.cdf((94.5-95)/0.25) * 2

"""
To what level would the supplier have to reduce the standard deviation of the operation 
if his goal were to obtain a capability score of 1 
(i.e., get less than or equal to 0.27 percent defective)?
"""
print getSD(Lower_Specification_Level=94.5, Upper_Specification_Level=95.5, CapabilityScore=1)