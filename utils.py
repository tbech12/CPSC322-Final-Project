"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #3

Description: utils for jupternotebooks
"""

import numpy as np

def get_column(table, header, col_name):
    '''
        Given a table and header and column name returns the column
    '''
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_frequencies(table, header, col_name):
    '''
        Given a table and header and column name returns
        the frequnces of a column
    '''
    col = get_column(table, header, col_name)
    col.sort() # inplace
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)

    return values, counts # we can return multiple values in python
    # they are packaged into a tuple

def group_by(table, header, groupby_col_name):
    '''
        Given a table and header and column name
        returns a group by
    '''
    groupby_col_index = header.index(groupby_col_name) # use this later
    groupby_col = get_column(table, header, groupby_col_name)
    group_names = sorted(list(set(groupby_col))) # e.g. [75, 76, 77]
    group_subtables = [[] for _ in group_names] # e.g. [[], [], []]

    for row in table:
        groupby_val = row[groupby_col_index] # e.g. this row's modelyear
        # which subtable does this row belong?
        groupby_val_subtable_index = group_names.index(groupby_val)
        group_subtables[groupby_val_subtable_index].append(row.copy()) # make a copy

    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    '''
        Given values and number of binds
        calculates the number of cutt offs
    '''
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # since bin_width is a float, we shouldn't use range() to generate a list
    # of cutoffs, use np.arange()
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exactly the max(values)
    # to handle round off error...
    # if your application allows, we should convert to int
    # or optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs

def compute_bin_frequencies(values, cutoffs):
    '''
        Given values and cutoffs compute the bins frequnces
    '''
    freqs = [0 for _ in range(len(cutoffs) - 1)] # because N + 1 cutoffs

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # add one to the last bin count
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1
                    # add one to this bin defined by [cutoffs[i], cutoffs[i+1])
    return freqs

def convert_to_string(data, col_identifier):
    '''
        Given a table and column will convert column to string
    '''
    index = data.column_names.index(col_identifier)
    for row in data.data:
        try:
            the_str = str(row[index])
            row[index] = the_str
        except ValueError:
            pass

def get_total(table, col_identifier):
    '''
        Given a table and column name
        gets the total of the column
    '''
    index = table.column_names.index(col_identifier)
    total = 0
    for row in table.data:
        if isinstance(row, str) is False:
            total += row[index]
    return total

def compute_mpg_rating(table, column_identifier):
    '''
       Given a table and column identifier
       Will calcuate the mpg ratings
    '''
    mpg_rating_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    mpg_rating_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    index = table.column_names.index(column_identifier)
    for row in table.data:
        if isinstance(row[index], (int, float)):
            if row[index] > 44.0:
                index_of_rating = mpg_rating_list.index(10)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 36.0 and row[index] <= 44.0:
                index_of_rating = mpg_rating_list.index(9)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 30.0 and row[index] <= 36.0:
                index_of_rating = mpg_rating_list.index(8)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 26.0 and row[index] <= 30.0:
                index_of_rating = mpg_rating_list.index(6)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 23.0 and row[index] <= 26.0:
                index_of_rating = mpg_rating_list.index(6)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 19.0 and row[index] <= 23.0:
                index_of_rating = mpg_rating_list.index(5)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 16.0 and row[index] <= 19.0:
                index_of_rating = mpg_rating_list.index(4)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] > 14.0 and row[index] <= 16.0:
                index_of_rating = mpg_rating_list.index(3)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] == 14.0:
                index_of_rating = mpg_rating_list.index(2)
                mpg_rating_count[index_of_rating] += 1
            elif row[index] <= 13.0:
                index_of_rating = mpg_rating_list.index(1)
                mpg_rating_count[index_of_rating] += 1
    return mpg_rating_list, mpg_rating_count

def compute_slope_intercept(x_value, y_value):
    '''
       Given an x and y calucates the intercept
    '''
    meanx = np.mean(x_value)
    meany = np.mean(y_value)

    num = sum([(x_value[i] - meanx) * (y_value[i] - meany) for i in range(len(x_value))])
    den = sum([(x_value[i] - meanx) ** 2 for i in range(len(x_value))])
    slope_of_the_line = num / den
    intercept = meany - slope_of_the_line * meanx
    return slope_of_the_line, intercept

def compute_coefficient_covariance(column_x_data, column_y_data):
    '''
        Given x data and y data computes the covariance and coefficent
    '''
    mean_column_x = sum(column_x_data) / len(column_x_data)
    mean_column_y = sum(column_y_data) / len(column_y_data)

    numerator = sum([(column_x_data[i] - mean_column_x) * (column_y_data[i] - mean_column_y)
                     for i in range(len(column_x_data))])
    denominator = (sum([(column_x_data[i] - mean_column_x) ** 2
                    for i in range(len(column_x_data))]) *
                    sum([(column_y_data[i] - mean_column_y) ** 2
                    for i in range(len(column_y_data))])) ** (1/2)
    coefficient = numerator / denominator
    covarient = numerator / (len(column_x_data) + len(column_y_data))
    return coefficient, covarient

def find_unique_values(data, columns, column_identifier):
    '''
        Given a table and header and column name retunrs all unique values
    '''
    genre_columns = get_column(data, columns, column_identifier)
    unique_values = []
    for value in genre_columns:
        if value != '':
            list_of_genres = value.split(',')
            for genre in list_of_genres:
                if genre not in unique_values:
                    unique_values.append(genre)
    return unique_values

def calculate_imdb_ratings(table):
    '''
        Given a table calcuates the IMDb ratings
    '''
    index = 5
    ratings = []
    for row in table:
        if isinstance(row[index], str):
            pass
        else:
            ratings.append(row[index])
    return ratings

def calculate_rt_ratings(table):
    '''
        Given a table calcuates the RT ratings
    '''
    rt_ratings = []
    for row in table:
        if row[6] != '':
            rt_ratings.append(float(row[6][:-1]))
    return rt_ratings

def calculate_genre_ratings(self, col_identifier, genre):
    '''
        Given a table calcuates the genre ratings
    '''
    index = self.column_names.index(col_identifier)
    table = []
    for row in self.data:
        if genre in row[index]:
            table.append(row)
    return table
