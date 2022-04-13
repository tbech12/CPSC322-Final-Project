"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #3

Description: plot utils for jupternotebooks
"""
import matplotlib.pyplot as plt
import utils
plt.style.use('seaborn-dark')

def bar_chart(values, columns, title, x_axis_name, y_axis_name):
    '''
        Function to plot a barchart
    '''
    plt.figure(figsize=(30,10))
    plt.title(title, size = 30)
    plt.xlabel(x_axis_name, size = 30)
    plt.ylabel(y_axis_name, size = 30)
    plt.bar(values, columns, align="edge", width=.4)
    plt.xticks(size = 20, rotation=90)
    plt.yticks(size = 20)
    plt.show()

def pie_chart(nums, the_labels, title):
    '''
        Function to plot a piechart
    '''
    plt.figure()
    plt.title(title, size = 30)
    plt.pie(nums, labels=the_labels, autopct="%1.1f%%", normalize=False)
    plt.show()

def hist_chart_frequency(values, columns, title, x_axis_name, y_axis_name):
    '''
        Function to plot a hist chart with freq
    '''
    plt.figure(figsize=(30,10))
    plt.title(title, size = 30)
    plt.xlabel(x_axis_name, size = 30)
    plt.ylabel(y_axis_name, size = 30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.bar(values[:-1], columns, width=(values[1] - values[0]), edgecolor="black", align="edge")
    plt.show()

def hist_chart(column_data, title, x_axis_name, y_axis_name):
    '''
        Function to plot a hist chart
    '''
    plt.figure(figsize=(30,10))
    plt.title(title, size = 30)
    plt.xlabel(x_axis_name, size = 30)
    plt.ylabel(y_axis_name, size = 30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.hist(column_data, bins=10)
    plt.xticks(rotation=90)
    plt.show()

def scatter_plot(column_x_data, column_y_data, title, x_axis_name, y_axis_name):
    '''
        Function to plot a scatter chart
    '''
    plt.figure()
    plt.title(title, size = 30)
    plt.xlabel(x_axis_name, size = 30)
    plt.ylabel(y_axis_name, size = 30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.scatter(column_x_data, column_y_data)
    plt.show()

def scatter_with_linear_regression(column_x_data, column_y_data, title, x_axis_name, y_axis_name):
    '''
        Function to plot a scatter chart with line for regression
    '''
    plt.figure()
    plt.title(title, size = 30)
    plt.xlabel(x_axis_name, size = 30)
    plt.ylabel(y_axis_name, size = 30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.scatter(column_x_data, column_y_data)
    coefficient, covariance = utils.compute_coefficient_covariance(column_x_data, column_y_data)
    slope_of_the_line, intercept = utils.compute_slope_intercept(column_x_data, column_y_data)
    plt.annotate("coefficient =" + str(coefficient), xy=(0.7,0.9),
                 xycoords="axes fraction", horizontalalignment="center")
    plt.annotate("covarient =" + str(covariance), xy=(0.7,0.8),
                 xycoords="axes fraction", horizontalalignment="center")
    plt.plot([min(column_x_data), max(column_x_data)],
             [slope_of_the_line * min(column_x_data) + intercept,
              slope_of_the_line * max(column_x_data) + intercept], c="r")

def box_plot(data, names, title, y_axis_name):
    '''
        Function to plot a box chart
    '''
    plt.figure(figsize=(30,10))
    plt.title(title, size = 30)
    plt.boxplot(data)
    plt.ylabel(y_axis_name, size = 30)
    plt.xticks(list(range(1, len(names) + 1)), names)
    plt.xticks(size = 20, rotation=45)
    plt.annotate(r"$\mu=100$", xy=(0.5, 0.5), xycoords="axes fraction",
                 horizontalalignment="center", color="blue")
    plt.show()
