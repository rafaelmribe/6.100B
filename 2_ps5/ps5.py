# -*- coding: utf-8 -*-
# Problem Set 5: Modeling Temperature Change
# Name: Rafael Moreno Ribeiro
# Collaborators: None

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAIN_INTERVAL = range(1961, 2000)
TEST_INTERVAL = range(2000, 2017)

##########################
#    Begin helper code   #
##########################

def standard_error_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by a linear
            regression model
        model: a numpy array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


class Dataset(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Dataset instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature

        f.close()

    def get_daily_temps(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d numpy array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_temp_on_date(self, city, month, day, year):
        """
        Get the temperature for the given city at the specified date.

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified date and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year {} is not available".format(year)
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

##########################
#    End helper code     #
##########################

    def calculate_annual_temp_averages(self, cities, years):
        """
        For each year in the given range of years, computes the average of the
        annual temperatures in the given cities.

        Args:
            cities: a list of the names of cities to include in the average
                annual temperature calculation
            years: a list of years to evaluate the average annual temperatures at

        Returns:
            a 1-d numpy array of floats with length = len(years). Each element in
            this array corresponds to the average annual temperature over the given
            cities for a given year.
        """
        # NOTE: TO BE IMPLEMENTED IN PART 4B.2 OF THE PSET
        # Initiate list to contain average temperatures
        temp_avg = []

        # Iterate through every city and year
        for city in cities:
            for year in years:
                # Get daily temperature array for given city and year
                daily_temps = self.get_daily_temps(city, year)

                # Get average temperature for city over given year
                annual_avg = float(np.mean(daily_temps))

                # Append values to list until list has first values for each year
                if len(temp_avg) != len(years):
                    temp_avg.append(annual_avg/(len(cities)))

                # After list already has first values, begin adding new values to the ones already present
                elif len(temp_avg) == len(years):
                    temp_avg[year - years[0]] += annual_avg/(len(cities))

        # Convert list into numpy array and return it
        return np.array(temp_avg)

def linear_regression(x, y):
    """
    Calculates a linear regression model for the set of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        (m, b): A tuple containing the slope and y-intercept of the regression line,
                both of which are floats.
    """
    num = 0
    den = 0

    avg_x = np.average(x)
    avg_y = np.average(y)


    for i in range(len(x)):
        num += (x[i]-avg_x)*(y[i]-avg_y)
        den += (x[i]-avg_x)**2

    m = num/den
    b = avg_y - m*avg_x

    return (m, b)

def squared_error(x, y, m, b):
    """
    Calculates the squared error of the linear regression model given the set
    of data points.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        m: The slope of the regression line
        b: The y-intercept of the regression line


    Returns:
        a float for the total squared error of the regression evaluated on the
        data set
    """
    SE = sum((y[i] - (m*x[i]+b))**2 for i in range(len(x)))
    return SE

def generate_polynomial_models(x, y, degrees):
    """
    Generates a list of polynomial regression models with degrees specified by
    degrees for the given set of data points

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        degrees: a list of integers that correspond to the degree of each polynomial
            model that will be fit to the data

    Returns:
        a list of numpy arrays, where each array is a 1-d numpy array of coefficients
        that minimizes the squared error of the fitting polynomial

        The models should appear in the list in the same order as their corresponding
        integers in the `degrees` parameter
    """
    return [np.polyfit(x, y, i, rcond=None) for i in degrees]

def evaluate_models(x, y, models, display_graphs=False):
    """
    For each regression model, compute the R-squared value for this model and
    if display_graphs is True, plot the data along with the best fit curve. You should make a separate plot for each model.

    Your plots should adhere to the following guidelines:

        - Plot the data points as individual blue (color='C10') dots.
        - Plot the model with a red (color='C3') solid line.
        - Include a title. Your title should include the $R^2$ value of the model and the degree. If the model is a linear curve (i.e. its degree is one), the title should also include the ratio of the standard error of this fitted curve's slope to the slope. Round your $R^2$ and SE/slope values to 4 decimal places.
        - Label the axes. You may assume this function will only be used in the case where the x-axis represents years and the y-axis represents temperature in degrees Celsius.

    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the R-squared value for each model
    """
    # Initiate list to contain R2 values
    R2_list = []

    # Iterate through every model
    for model in models:
        # Calculate R2 value and append it to the list
        y_estimates = np.polyval(model, x)
        R2 = r2_score(y, y_estimates)
        R2_list.append(R2)

        if display_graphs == True:
            # Create plot fo contain the results
            plt.figure(figsize=(12, 8))

            # Scatter data points
            plt.scatter(x, y, color='C10')

            # Plot model line
            plt.plot(x, np.polyval(model, x), label='Fitted curve', color='C3', linestyle='solid')

            plt.xlabel('Year')
            plt.ylabel('Temperature (°C)')
            if len(model) == 2:
                plt.title(f'Change in temperature over time in the U.S.\nModel degree = {len(model)-1} | R{chr(0xB2)} = {round(R2, 4)} | SE/slope = {round(standard_error_over_slope(x, y, y_estimates, model), 4)}')
            else:
                plt.title(f'Change in temperature over time in the U.S.\nModel degree = {len(model)-1} | R{chr(0xB2)} = {round(R2, 4)}')
            plt.legend()
            plt.show()
        else:
            pass

    return R2_list


def get_max_trend(x, y, length, positive_slope):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points
        length: the length of the interval
        positive_slope: a boolean whose value specifies whether to look for
            an interval with the most extreme positive slope (True) or the most
            extreme negative slope (False)

    Returns:
        a tuple of the form (i, j, m) such that the application of linear (deg=1)
        regression to the data in x[i:j], y[i:j] produces the most extreme
        slope m, with the sign specified by positive_slope and j-i = length.

        In the case of a tie, it returns the first interval. For example,
        if the intervals (2,5) and (8,11) both have slope 3.1, (2,5,3.1) should be returned.

        If no intervals matching the length and sign specified by positive_slope
        exist in the dataset then return None
    """
    # Initiate counter to go through every value in x
    i = 0
    # Variable to track i of the maximum m found so far
    best_i_so_far = None
    # Variable to track maximum m found so far
    max_m_so_far = 0

    # Loop through every entry of x
    while i <= len(x) - length:
        # Define short intervals in x and y
        small_x = x[i:i+length]
        small_y = y[i:i+length]

        # Find coefficients and store them as m and b
        m, b = linear_regression(small_x, small_y)

        # Test if new m is better then max_m_so_far and update value of our variables if true
        if (positive_slope and m > max_m_so_far) or (not positive_slope and m < max_m_so_far):
            max_m_so_far = m
            best_i_so_far = i
        else:
            pass

        # Go to next data point
        i += 1

    if best_i_so_far == None:
        return None

    else:
        return (best_i_so_far, best_i_so_far+length, max_m_so_far)

    # What about the 1e-8 thing??


def get_all_max_trends(x, y):
    """
    Args:
        x: a 1-d numpy array of length N, representing the x-coordinates of
            the N sample points
        y: a 1-d numpy array of length N, representing the y-coordinates of
            the N sample points

    Returns:
        a list of tuples of the form (i,j,m) such that the application of linear
        regression to the data in x[i:j], y[i:j] produces the most extreme
        positive OR negative slope m, and j-i=length.

        The returned list should have len(x) - 1 tuples, with each tuple representing the
        most extreme slope and associated interval for all interval lengths 2 through len(x).
        If there is no positive or negative slope in a given interval length L (m=0 for all
        intervals of length L), the tuple should be of the form (0,L,None).

        The returned list should be ordered by increasing interval length. For example, the first
        tuple should be for interval length 2, the second should be for interval length 3, and so on.

        If len(x) < 2, return an empty list
    """
    tuple_list = []

    if len(x) < 2:
        return tuple_list

    for length in range(2, len(x) + 1):
        max_trend_pos = get_max_trend(x, y, length, True)
        max_trend_neg = get_max_trend(x, y, length, False)

        i_pos, j_pos, m_pos = max_trend_pos if max_trend_pos else (0, length, None)
        i_neg, j_neg, m_neg = max_trend_neg if max_trend_neg else (0, length, None)

        if m_pos is not None and (m_neg is None or abs(m_pos) > abs(m_neg)):
            tuple_list.append((i_pos, j_pos, m_pos))
        elif m_neg is not None:
            tuple_list.append((i_neg, j_neg, m_neg))
        else:
            tuple_list.append((0, length, None))

    return tuple_list


def calculate_rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d numpy array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    RMSE = np.sqrt(sum((y[i] - estimated[i])**2 for i in range(len(y)))/len(y))
    return RMSE

def evaluate_rmse(x, y, models, display_graphs=False):
    """
    For each regression model, compute the RMSE for this model and if
    display_graphs is True, plot the test data along with the model's estimation.

    For the plots, you should plot data points (x,y) as green (color='C2') dots and your best
    fit curve (aka model) as an orange (color='C1') solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points.

    RMSE should be rounded to 4 decimal places.

    Args:
        x: a 1-d numpy array with length N, representing the x-coordinates of
            the N test data sample points
        y: a 1-d numpy array with length N, representing the y-coordinates of
            the N test data sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a numpy array storing the coefficients of
            a polynomial.
        display_graphs: A boolean whose value specifies if the graphs should be
            displayed

    Returns:
        A list holding the RMSE value for each model
    """
    # Initiate list to contain R2 values
    RMSE_list = []

    # Iterate through every model
    for model in models:
        # Calculate R2 value and append it to the list
        y_estimates = np.polyval(model, x)
        RMSE = calculate_rmse(y, y_estimates)
        RMSE_list.append(round(RMSE, 4))

        if display_graphs == True:
            # Create plot fo contain the results
            plt.figure(figsize=(12, 8))

            # Scatter data points
            plt.scatter(x, y, color='C2')

            # Plot model line
            plt.plot(x, np.polyval(model, x), label='Fitted curve', color='C1', linestyle='solid')

            plt.xlabel('Year')
            plt.ylabel('Temperature (°C)')
            plt.title(f'Change in temperature over time in the U.S.\nModel degree = {len(model)-1} | RMSE = {round(RMSE, 4)}')
            plt.legend()
            plt.show()
        else:
            pass

    return RMSE_list


if __name__ == '__main__':
    pass
    ##################################################################################
    # Problem 4A: DAILY TEMPERATURE
    data_set = Dataset('data.csv')
    x = np.arange(1961, 2017)
    y = np.array(list(data_set.get_temp_on_date('PHOENIX', 12, 1, year) for year in range(1961, 2017)))
    models = generate_polynomial_models(x, y, [1])
    evaluate_models(x, y, models, display_graphs=True)
    '''plt.savefig('4A.png')'''

    ##################################################################################
    # Problem 4B: ANNUAL TEMPERATURE
    '''x = np.arange(1961, 2017)
    y = np.array(data_set.calculate_annual_temp_averages(['PHOENIX'], range(1961, 2017)))
    models = generate_polynomial_models(x, y, [1])
    evaluate_models(x, y, models, display_graphs=True)
    plt.savefig('4B.png')'''

    ##################################################################################
    # Problem 5B: INCREASING TRENDS
    '''x = np.arange(1961, 2017)
    y = np.array(data_set.calculate_annual_temp_averages(['SEATTLE'], range(1961, 2017)))
    i, j, m = get_max_trend(x, y, 30, True)
    models = generate_polynomial_models(x[i:j], y[i:j], [1])
    evaluate_models(x[i:j], y[i:j], models, display_graphs=True)
    plt.savefig('5B.png')'''

    ##################################################################################
    # Problem 5C: DECREASING TRENDS
    '''x = np.arange(1961, 2017)
    y = np.array(data_set.calculate_annual_temp_averages(['SEATTLE'], range(1961, 2017)))
    i, j, m = get_max_trend(x, y, 12, False)
    models = generate_polynomial_models(x[i:j], y[i:j], [1])
    evaluate_models(x[i:j], y[i:j], models, display_graphs=True)
    plt.savefig('5C.png')'''

    ##################################################################################
    # Problem 5D: ALL EXTREME TRENDS
    # Your code should pass test_get_max_trend. No written answer for this part, but
    # be prepared to explain in checkoff what the max trend represents.

    ##################################################################################
    # Problem 6B: PREDICTING
    # Part 1
    '''x_train = np.arange(1961, 2000)
    y_train = np.array(data_set.calculate_annual_temp_averages(CITIES, range(1961, 2000)))
    models = generate_polynomial_models(x_train, y_train, [2, 10])
    evaluate_models(x_train, y_train, models, display_graphs=True)
    plt.savefig('6B_1.png')'''

    #Part 2
    '''x_test = np.arange(2000, 2017)
    y_test = np.array(data_set.calculate_annual_temp_averages(CITIES, range(2000, 2017)))
    evaluate_rmse(x_test, y_test, models, display_graphs=True)
    plt.savefig('6B_2.png')'''
    ####################################################################################
