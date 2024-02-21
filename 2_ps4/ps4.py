# -*- coding: utf-8 -*-
# 6.100B Fall 2023
# Problem Set 4: Sea Level Rise
# Name: Rafael Moreno Ribeiro
# Collaborators: Jair Santana

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scipy.stats as st
from scipy.interpolate import interp1d

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 97.5th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 97.5th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper - mean) / st.norm.ppf(.975)

def load_data():
    """
	Loads data from sea_level_change.csv and puts it into numpy arrays

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year', 'Lower', 'Upper']
    return (df.Year.to_numpy(), df.Lower.to_numpy(), df.Upper.to_numpy())


###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100, inclusive, and not included in the data,
    the values for that year should be interpolated. If show_plot, displays a
    plot with mean and the 95% confidence interval, assuming sea level rise
    follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing the year, the mean, the 2.5th
        percentile, 97.5th percentile, and standard deviation of the sea level rise
        for the years between 2020-2100 inclusive
	"""
    years, slow_slr, fast_slr = load_data()

    # Interpolate to estimate slow and fast SLR values for the missing years
    slow_interp = interp1d(years, slow_slr, kind='linear', fill_value='extrapolate')
    fast_interp = interp1d(years, fast_slr, kind='linear', fill_value='extrapolate')

    all_years = np.arange(2020, 2101)
    slow_slr_interp = slow_interp(all_years)
    fast_slr_interp = fast_interp(all_years)

    # Find mean, 2.5th percentile, 97.5th percentile, and standard deviation
    mean_slr = (slow_slr_interp + fast_slr_interp) / 2
    std_slr = calculate_std(fast_slr_interp, mean_slr)

    # Initiate 2D numpy array with the required information
    sea_level_data = np.column_stack((all_years, mean_slr, slow_slr_interp, fast_slr_interp, std_slr))

    # Plot only if show_plot is True
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(all_years, fast_slr_interp, label='Upper bound', color='b', linestyle='--')
        plt.plot(all_years, mean_slr, label='Mean', color='r')
        plt.plot(all_years, slow_slr_interp, label='Lower bound', color='g', linestyle='--')
        plt.xlabel('Year')
        plt.ylabel('Sea Level Rise (SLR)')
        plt.title('Predicted Sea Level Rise (2020-2100)')
        plt.legend()
        plt.show()

    return sea_level_data


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

    (This function should use the predicted_sea_level_rise function.)

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    # Use np.where to find the index of the specified year
    index = np.where(data[:, 0] == year)[0][0]

    # Find mean and standard deviation for that year
    mean_slr = data[index, 1]
    std_slr = data[index, 4]

    # Simulate sea level rise using a normal distribution
    simulated_data = np.random.normal(loc=mean_slr, scale=std_slr, size=num)

    return simulated_data


def plot_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    years, mean, lower, upper, _ = data.T

    # Set up the plot
    plt.figure(figsize=(12, 6))
    plt.plot()
    plt.xlabel('Year')
    plt.ylabel('Simulated Sea Level Rise')
    plt.title('Monte Carlo Simulation of Sea Level Rise')

    # Plot the Monte Carlo simulations for each year
    for year_idx, year in enumerate(years):
        simulations = simulate_year(data, year, 500)

        # Plot individual points for each simulation
        plt.scatter([year] * 500, simulations, color='gray', alpha=0.1, s=10)

    # Plot the mean line for each year
    plt.plot(years, upper, '--', color='b', label='Upper Bound', linewidth=1)
    plt.plot(years, mean, color='tab:orange', label='Mean', linewidth=2)
    plt.plot(years, lower, '--', color='g', label='Lower Bound', linewidth=1)

    plt.legend()
    plt.show()


##########
# Part 2 #
##########

def simulate_water_levels(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a python list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    # Initiate list ro contain simulation data
    simulated_water_levels = []

    # Loop through each year in the range 2020 to 2100
    for year in range(2020, 2101):
        # Simulate water level for the current year using simulate_year
        simulated_data = simulate_year(data, year, num=1)
        # Append the simulated value to the list
        simulated_water_levels.append(simulated_data[0])

    return simulated_water_levels


def repair_only(water_level_list, water_level_loss_no_prevention, house_value=500000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		a python list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # Initiate list to contain costs
    costs = []

    for water_level in water_level_list:
        if water_level <= 5:
            costs.append(0)
        elif water_level < 10:
            # Interpolate the percentage for non-integer water levels
            interpolated_percentage = np.interp(water_level, water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1])
            cost = house_value * (interpolated_percentage / 100)
            costs.append(cost/1000)
        else:
            costs.append(house_value/1000)

    return costs


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=500000,
               cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having two consecutive years with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have two consecutive years with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # Initialize variables for consecutive excessive cost years and flood prevention implementation
    consecutive_excessive_years = 0
    flood_prevention_activated = False
    damage_costs = []

    for water_level in water_level_list:
        # Calculate damage cost based on the strategy
        if water_level <= 5:
            cost = 0
        elif 5 < water_level < 10:
            if flood_prevention_activated:
                percentage = np.interp(water_level, water_level_loss_with_prevention[:, 0], water_level_loss_with_prevention[:, 1])
            else:
                percentage = np.interp(water_level, water_level_loss_no_prevention[:, 0], water_level_loss_no_prevention[:, 1])

            cost = house_value * percentage / 100

            # Check if cost exceeds the threshold for consecutive excessive years
            if cost > cost_threshold:
                consecutive_excessive_years += 1
            else:
                consecutive_excessive_years = 0

            # Check if flood prevention measures should be implemented
            if consecutive_excessive_years >= 2:
                flood_prevention_activated = True

        else:
            cost = house_value

        damage_costs.append(cost/1000)

    return damage_costs



def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=500000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft (exclusive), the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # Extract SLR levels and corresponding property damage percentages
    slr_levels_with_prevention = water_level_loss_with_prevention[:, 0]
    property_damage_percentages_with_prevention = water_level_loss_with_prevention[:, 1]

    # Use interpolation to estimate property damage percentages for simulated water levels
    property_damage_interp_with_prevention = interp1d(slr_levels_with_prevention, property_damage_percentages_with_prevention, kind='linear', fill_value='extrapolate')
    interpolated_percentages_with_prevention = property_damage_interp_with_prevention(water_level_list)

    # Calculate damage costs based on the prepare immediately strategy
    damage_costs = []

    for i in range(len(water_level_list)):
        percentage_with_prevention = interpolated_percentages_with_prevention[i]

        if water_level_list[i] <= 5:
            # If the water level is <= 5ft, the cost is 0.
            cost = 0
        elif 5 < water_level_list[i] < 10:
            # If the water level is between 5ft and 10ft, find cost from on house value and given percentage
            cost = house_value * (percentage_with_prevention / 100)
        else:
            # If the water level is bigger than or equal to 10ft, the cost is the entire value of the house.
            cost = house_value

        # Append the cost to the list
        damage_costs.append(cost/1000)

    return damage_costs


def plot_prep_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=500000,
                    cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 2.5th percentile, 97.5th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    years, lower, upper, mean, std = data.T

    # Run 500 simulations for each year for each strategy
    simulations_repair_only = []
    simulations_wait_a_bit = []
    simulations_prepare_immediately = []

    for year in years:
        water_levels = simulate_year(data, year, 500)

        # Calculate damage costs for each strategy
        costs_repair_only = repair_only(water_levels, water_level_loss_no_prevention, house_value)
        costs_wait_a_bit = wait_a_bit(water_levels, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
        costs_prepare_immediately = prepare_immediately(water_levels, water_level_loss_with_prevention, house_value)

        # Append the costs to the respective lists
        simulations_repair_only.append(costs_repair_only)
        simulations_wait_a_bit.append(costs_wait_a_bit)
        simulations_prepare_immediately.append(costs_prepare_immediately)

    # Calculate mean for each strategy
    mean_repair_only = np.mean(simulations_repair_only, axis=1)
    mean_wait_a_bit = np.mean(simulations_wait_a_bit, axis=1)
    mean_prepare_immediately = np.mean(simulations_prepare_immediately, axis=1)


    # Create plot fo contain the results
    plt.figure(figsize=(12, 8))

    # Scatter plot for each strategy
    plt.scatter(np.repeat(years, 500), np.array(simulations_repair_only).flatten(), alpha=0.5, s=5)
    plt.scatter(np.repeat(years, 500), np.array(simulations_wait_a_bit).flatten(), alpha=0.5, s=5)
    plt.scatter(np.repeat(years, 500), np.array(simulations_prepare_immediately).flatten(), alpha=0.5, s=5)

    # Line plot for mean damage losses
    plt.plot(data[:, 0], mean_repair_only, label='Repair-only scenario', color='blue', linestyle='-')
    plt.plot(data[:, 0], mean_wait_a_bit, label='Wait a bit scenario', color='green', linestyle='-')
    plt.plot(data[:, 0], mean_prepare_immediately, label='Prepare immediately scenario', color='orange', linestyle='-')


    plt.xlabel('Year')
    plt.ylabel('Damage Costs ($K)')
    plt.title('Monte Carlo Simulation of Preparation Strategies')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Comment out the 'pass' statement below to run the lines below it
    pass

    # Uncomment the following lines to plot generate plots
    #data = predicted_sea_level_rise(show_plot=False)
    #water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    #water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    #plot_simulation(data)
    #plot_prep_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
