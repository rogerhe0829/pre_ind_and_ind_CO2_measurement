import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.optimize import curve_fit

# Process the given csv, return arrays based on its header row.
def read_file(filename):
    """Return five numpy arrays named year,temp_mean,temp_std,co2_mean,and co2_std respectively
    given the data from the file filename.
    """
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, filling_values=np.nan) # originally used np.loadtxt(), but unable to convert empty string in to valid output
    year, temp_mean, temp_std, co2_mean, co2_std = data.T
    return year, temp_mean, temp_std, co2_mean, co2_std


# Equation of estimated temperature given the CO2 amount.
def log_func(co2_mean, a, b) -> float:
    """Return the estimated change in temperature given the CO2 amount co2_amount based on the equation:
    delta t = a * log(C) + b, where delta t is the change in temperature, C is the CO2 amount,and a, b
    are unknown parameters
    """

    delta_t = a * np.log(co2_mean) + b
    return delta_t


# Function which returns the mean value of a specific numpy array dataset
def find_mean(dataset: np.ndarray) -> float:
    """Return the mean value of the given dataset.
    """

    mean = np.nanmean(dataset)
    return mean


# Function which returns the standard deviation of a specific numpy array dataset
def find_standard_deviation(dataset: np.ndarray) -> float:
    """Return the standard deviation of the given dataset.
    """

    mean = np.sum(dataset) / len(dataset)
    accumulator = 0
    for i in range(len(dataset)):
        accumulator += (dataset[i] - mean) ** 2
    std = np.sqrt(accumulator / len(dataset))
    return std


# ------------------------------------------------
# Pre-industrial period (1750-1850), index 752 - 852

# histogram for co2
def co2_histogram(co2_mean):
    """Generate a histogram given the carbon dioxide dataset co2_mean
    in the pre-industrial and industrial time period.
    """

    pre_min_range_c02 = min(co2_mean[:772])
    pre_max_range_c02 = max(co2_mean[:772])
    post_min_range_c02 = min(co2_mean[772:])
    post_max_range_c02 = max(co2_mean[772:])
    plt.hist(co2_mean[:772], bins = 20, range = (pre_min_range_c02, pre_max_range_c02), density = True, label='Pre-Industrial CO2 Amount')
    plt.hist(co2_mean[772:], bins = 20, range=(post_min_range_c02, post_max_range_c02), density=True, label='Industrial CO2 Amount')
    plt.legend()
    plt.xlabel('CO2 Amount')
    plt.ylabel('Frequency')
    plt.title('CO2 Amount Distribution')
    plt.show()


# histogram for temperature
def temp_histogram(temp_mean):
    """Generate a histogram given the temperature dataset temp_mean
    in the pre-industrial and industrial time period.
    """

    # Plotting
    pre_min_temp = min(temp_mean[:772])
    pre_max_temp = max(temp_mean[:772])
    post_min_temp = min(temp_mean[772:])
    post_max_temp = max(temp_mean[772:])
    plt.hist(temp_mean[:772], bins = 20, range = (pre_min_temp, pre_max_temp), density = True, label='Pre-Industrial Temperature Temperature')
    plt.hist(temp_mean[772:], bins = 20, range=(post_min_temp, post_max_temp), density=True, label='Industrial Temperature')
    plt.legend()
    plt.xlabel('Temperature')
    plt.ylabel('Frequency')
    plt.title('Temperature Distribution')
    plt.show()


# Regression Plot
def regression(co2_mean, temp_mean):
    """Generate a plot with regression line and residuals givcen the CO2 value co2_mean
    and temperature values temp_mean.
    """

    # Check validity
    valid_indices = np.isfinite(co2_mean) & np.isfinite(temp_mean)  # Get valid indices
    valid_co2 = co2_mean[valid_indices]
    valid_temp = temp_mean[valid_indices]

    # Regression line
    params, covariance = curve_fit(log_func, valid_co2, valid_temp)
    a, b = params

    # plotting
    plt.scatter(np.log(valid_co2), valid_temp, label='Observed Data', color='blue')
    plt.plot(np.log(valid_co2), log_func(valid_co2, *params), color='red', label='Fitted Line')
    plt.title('Original Temperature Data and Predicted Temperature Data')
    plt.xlabel('log(CO2)')
    plt.ylabel('Temperature Change')
    plt.legend()
    plt.grid()
    plt.show()


# Residual plot
def residual_plot(co2_mean, temp_mean):
    """Generate a plot of residuals given the original data co2_mean and temp_mean.
    """

    # Check validity
    valid_indices = np.isfinite(co2_mean) & np.isfinite(temp_mean)  # Get valid indices
    valid_co2 = co2_mean[valid_indices]
    valid_temp = temp_mean[valid_indices]

    # Calculate predicted temperature based on the regression model
    params, covariance = curve_fit(log_func, valid_co2, valid_temp)
    predicted_temp = log_func(valid_co2, *params)

    # Calculate residuals
    residuals = valid_temp - predicted_temp

    # Residual plot
    plt.scatter(np.log(valid_co2), residuals, color='green')
    plt.axhline(0, color='purple', lw=1, ls='-')
    plt.title('Residual Plot')
    plt.xlabel('log(CO2)')
    plt.ylabel('Residuals')
    plt.grid()
    plt.show()

# Compare the distributions
def analyze_overlap(co2_mean, temp_mean):
    """Determine if there exists overlap in dataset between two periods.
     """

    # Overlap Condition: |pre_ind_mean - ind_mean| < (pre_in_std + ind_std)
    # CO2 computation
    diff_in_mean = abs(np.mean(co2_mean[:772]) - np.mean(co2_mean[:772]))
    sum_of_std = np.std(co2_mean[:772]) + np.std(co2_mean[772:])
    if diff_in_mean < sum_of_std:
        print('There exists overlap between CO2 distribution in the pre-industrial ',
              'and industrial periods.')
    else:
        print('There exists no overlap between CO2 distribution in the pre-industrial ',
              'and industrial periods.')


    # Temperature computation
    diff_in_mean = abs(np.mean(temp_mean[:772]) - np.mean(temp_mean[:772]))
    sum_of_std = np.std(temp_mean[:772]) + np.std(temp_mean[772:])
    if diff_in_mean < sum_of_std:
        print('There exists overlap between temperature distribution in the pre-industrial',
              'and industrial periods.')
    else:
        print('There exists no overlap between temperature distribution in the pre-industrial',
              'and industrial periods.')


# Find year where co2 and temperature are significantly different in two periods
def find_difference_year(co2_mean, temp_mean):
    """Determine if there is a significant difference in the dataset between two periods."""

    # Calculate mean and standard deviation for the pre-industrial period (up to year 772)
    pre_ind_co2_mean = np.mean(co2_mean[:772])
    pre_ind_co2_std = np.std(co2_mean[:772])
    pre_ind_temp_mean = np.mean(temp_mean[:772])
    pre_ind_temp_std = np.std(temp_mean[:772])

    co2_year_acc = []
    temp_year_acc = []

    # Load the data
    data = np.genfromtxt('co2_temp_1000.csv', delimiter=',', skip_header=1, filling_values=np.nan)
    year, temp_mean, temp_std, co2_mean, co2_std = data.T

    # Set the number of consecutive years required for a consistent deviation
    consecutive_threshold = 3
    co2_consecutive_count = 0
    temp_consecutive_count = 0

    # Find difference in CO2
    for i in range(772, len(year)):
        if co2_mean[i] > pre_ind_co2_mean + 2 * pre_ind_co2_std: # Difference condition: 2 standard deviation above the mean
            co2_consecutive_count += 1
        else:   # Difference Condition failed
            co2_consecutive_count = 0

        if co2_consecutive_count >= consecutive_threshold:  # Accumulator to verify the validity of difference
            co2_year_acc.append(year[i - consecutive_threshold + 1])
            break  # Stop after finding the first deviation

    if not co2_year_acc:
        print('There exists no significant difference in CO2 between two periods.')
    else:
        print(f'There exists a significant difference in CO2 between two periods, starting in year: {co2_year_acc[0]}', 'that is 1830s')

    # Find difference in temperature
    for i in range(772, len(year)):
        if temp_mean[i] > pre_ind_temp_mean + 2 * pre_ind_temp_std:     # Difference condition: 2 standard deviation away from the mean
            temp_consecutive_count += 1
        else:       # Difference condition failed
            temp_consecutive_count = 0  # Reset if it doesn't exceed the threshold

        if temp_consecutive_count >= consecutive_threshold:     # accumulator which verifies the validity of the difference
            temp_year_acc.append(year[i - consecutive_threshold + 1])
            break  # Stop after finding the first deviation

    if not temp_year_acc:
        print('There exists no significant difference in temperature between two periods.')
    else:
        print(
            f'There exists a significant difference in temperature between two periods, starting in year: {temp_year_acc[0]}', 'that is 1920s')


# Find predicted change in temperature
def find_change_in_temp(co2_mean, temp_mean):
    """Return the predicted change in temperature given the CO2 value co2_mean.
    """

    valid_indices = np.isfinite(co2_mean) & np.isfinite(temp_mean)
    valid_co2 = co2_mean[valid_indices]
    valid_temp = temp_mean[valid_indices]
    params, covariance = curve_fit(log_func, valid_co2, valid_temp)
    a, b = params
    predicted_temp_change = log_func(2 * co2_mean, *params)
    return predicted_temp_change


"""
# Find significant difference of dataset between two periods
def find_difference_year(co2_mean, temp_mean):
    Determine if there is significant difference of the dataset between two periods.
    

    pre_ind_co2_mean = np.mean(co2_mean[:772])
    pre_ind_co2_std = np.std(co2_mean[:772])
    pre_ind_temp_mean = np.mean(temp_mean[:772])
    pre_ind_temp_std = np.std(temp_mean[:772])
    co2_year_acc = []
    temp_year_acc = []

    # Year Data
    data = np.genfromtxt('co2_temp_1000.csv', delimiter=',', skip_header=1, filling_values=np.nan) # originally used np.loadtxt(), but unable to convert empty string in to valid output
    year, temp_mean, temp_std, co2_mean, co2_std = data.T

    # Find difference in CO2
    for i in range(772, len(year)):
        if co2_mean[i] > pre_ind_co2_mean + 2 * pre_ind_co2_std:
            co2_year_acc.append(year[i])
    
    if co2_year_acc == []:
        print('There exists no significant difference in CO2 between two periods')
    else:
        print('There exists significant difference in CO2 between two periods, that is {}'.format(co2_year_acc))

    # Find difference in temperature
    for i in range(772, len(year)):
        if temp_mean[i] > pre_ind_temp_mean + 2 * pre_ind_temp_std:
            temp_year_acc.append(year[i])
    if temp_year_acc == []:
        print('There exists no significant difference in CO2 between two periods')
    else:
        print('There exists significant difference in CO2 between two periods, that is {}'.format(temp_year_acc))

"""


#def find_overlap(co2_mean, year):
#    """Return the years when pre-industrial and industrial CO2 and temperature
#    levels are the same of pre-industrial values.
#   """
#
#    years = []
#    for i in range(len(co2_mean[772-229:772]) - 1):
#        if ((co2_mean[772-229:772][i] >  co2_mean[772:][i] and co2_mean[772-229:772][i+1] <  co2_mean[772:][i+1]) or
#            (co2_mean[772-229:772][i] <  co2_mean[772:][i] and co2_mean[772-229:772][i+1] >  co2_mean[772:][i+1])):
#            years.append(((year[772-229:772][i], year[772-229:772][i+1]), (year[772:][i], year[772:][i+1])))
#    return years


# ------------------------------------------------

def main():
    year, temp_mean, temp_std, co2_mean, co2_std = read_file('co2_temp_1000.csv')
    # CO2 mean value for the pre-industrial period.
    pre_ind_co2_mean = find_mean(co2_mean[:772])
    print('The CO2 mean value of the pre-industrial period is {}'.format(pre_ind_co2_mean))

    # CO2 mean value for the industrial period.
    ind_co2_mean = find_mean(co2_mean[772:])
    print('The CO2 mean value of the pre-industrial period is {}'.format(ind_co2_mean))

    # Temperature mean value for the pre-industrial period
    pre_ind_temp_mean = find_mean(temp_mean[:772])
    print('The temperature mean value of the pre-industrial period is {}'.format(pre_ind_temp_mean))

    # Temperature mean value for the industrial period
    ind_temp_mean = find_mean(temp_mean[772:])
    print('The temperature mean value of the industrial period is {}'.format(ind_temp_mean))

    # Find CO2 standard deviation of the pre-industrial period
    pre_ind_co2_std = find_standard_deviation(co2_mean[:772])
    print('The CO2 standard deviation of the pre-industrial period is {}'.format(pre_ind_co2_std))

    # Find CO2 standard deviation of the industrial period
    ind_co2_std = find_standard_deviation(co2_mean[772:])
    print('The CO2 standard deviation of the industrial period is {}'.format(ind_co2_std))

    # Find temperature standard deviation of the pre-industrial period
    pre_ind_temp_std = find_mean(temp_mean[:772])

    # Return the predicted change in temperature
    change_in_temp = find_change_in_temp(co2_mean, temp_mean)
    print('The predicted change in temperature when the CO2 is doubled is {}'.format(change_in_temp))

    # CO2 histogram plot
    co2_histogram(co2_mean)

    # Temperature histogram plot
    temp_histogram(temp_mean)

    # Regression Plot
    regression(co2_mean, temp_mean)

    # Residual plot
    residual_plot(co2_mean, temp_mean)

    # Chck if there is overlap between CO2 and temperature in two periods
    overlap_condition = analyze_overlap(co2_mean, temp_mean)
    print(overlap_condition)

    # years of different co2 and temp in two periods
    diff_year = find_difference_year(co2_mean, temp_mean)
    print(diff_year)

    valid_indices = np.isfinite(co2_mean) & np.isfinite(temp_mean)  # Get valid indices
    valid_co2 = co2_mean[valid_indices]
    valid_temp = temp_mean[valid_indices]
    params, covariance = curve_fit(log_func, valid_co2, valid_temp)
    predicted_temp = log_func(valid_co2, *params)
    residuals = valid_temp - predicted_temp
    print("Length of valid_co2:", len(valid_co2))
    print("Length of residuals:", len(residuals))



if __name__ == '__main__':
    main()