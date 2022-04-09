"""
Programmer: Troy Bechtold
Class: CptS 322-01, Spring 2022
Programming Assignment #2
2/9/22
I attempted the bonus number 3

Description: This program is the begining of the
mypytable library we are building
"""
import os
import copy
import csv
from mysklearn import myutils
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        #gets len of data and column names
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        #gets col index
        col_index = self.column_names.index(col_identifier)
        columns = []
        #if set to false skips NA values
        if include_missing_values is False:
            for row in self.data:
                if row[col_index] != "NA":
                    columns.append(row[col_index])
        else: #else it will get every row
            for row in self.data:
                columns.append(row[col_index])
        return columns

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in range(len(self.data)): #loop through all rows
            for column in range(len(self.column_names)): #loop through all cols
                try:
                    numeric_value = float(self.data[row][column]) #try to convert the type
                    self.data[row][column] = numeric_value #saves to data
                except ValueError:
                    #print(self.data[row], " could not be converted to a numeric type")
                    pass #does error but continue on processing
        return self

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        row_len = len(self.data) #length of data
        count = 0
        for index in row_indexes_to_drop: #look at each row to drop
            if index < row_len: #if the index is less than max data row
                try:
                    self.data.pop(index - count) #pop the row
                    count += 1
                except IndexError:
                    pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
         # Open file
        infile = open(filename, "r")
        with open(filename, "r") as infile:#opens file
            csvreader = csv.reader(infile, delimiter = ",")
            first_row = next(csvreader)
            for row in first_row: # get the first row as the header
                self.column_names.append(row)
            for row in csvreader: #get the rest of the data as the table
                self.data.append(row)
            self.convert_to_numeric()
            infile.close() #close file
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        #print("Saving to: ", filename)
        out = open(filename, "w") #open file
        writer = csv.writer(out) #get csv writer
        writer.writerow(self.column_names) #right column
        writer.writerows(self.data) #right rows
        out.close() #close
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dups = []
        indexes = []
        rows = []
        for i in range(len(key_column_names)):
            #get all indexes of the column names
            indexes.append(self.column_names.index(key_column_names[i]))
        for row in self.data: #loop throygh data
            dup_row = []
            for i in indexes: #loop through indexes
                dup_row.append(row[i]) #add to dup row
            if dup_row in rows: #if found dupe in dupe dows
                dups.append(self.data.index(row)) #add to dup
            else:
                rows.append(dup_row) #else add to row
        return dups

    def remove_rows_with_missing_values(self):
        """
        Remove rows from the table data that contain a missing value ("NA").
        """
        count = 0
        newTable = []
        for row in range(len(self.data)): #loop through data
            for column in range(len(self.column_names)): #loop through cols
                if self.data[row][column] != "NA": #if not NA
                    count +=1
                else:
                    pass
                    #print("Removing: ", self.data[row])
                if count == len(self.column_names):
                    newTable.append(self.data[row]) #add valid rows
            count = 0
        self.data = []
        self.data = copy.deepcopy(newTable) #save valid rows to data

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        column_elements = []
        column_index = self.column_names.index(col_name) #get column index by given col
        for value in self.data:
            if value[column_index] != "NA": #if the value is not NA
                column_elements.append(value[column_index]) #add to column elements
        try:
            avg = sum(column_elements) / len(column_elements) #gets average
            for row in range(len(self.data)):
                if self.data[row][column_index] == "NA": #if the row is na
                    self.data[row][column_index] = avg #repalce with avg
        except ValueError:
            pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        curr_table = []
        col_indexes = []
        col_len = len(col_names) #get length of column names
        index = 0
        while index < col_len:
            for column in range(len(self.column_names)):
                if col_names[index] == self.column_names[column]: #if the column is in mypytables
                    col_indexes.append(column) #adds all columns
            index += 1
        if not self.data: #if no data
            index2 = 0
            curr_table = []
            while index2 < len(col_indexes):
                curr_stats = []
                curr_stats.append(self.column_names[col_indexes[index2]])
                curr_table.append(curr_stats)
                index2 += 1
            new_table = MyPyTable(curr_table, []) #send back attributes with not data
            return new_table#return data
        else:
            index2 = 0
            while index2 < len(col_indexes):
                curr_stats = []
                curr_stats.append(self.column_names[col_indexes[index2]])
                min = self.data[0][col_indexes[index2]] #get a temporary min
                for row in range(len(self.data)): #for the rows in data
                    if self.data[row][col_indexes[index2]] == "NA": #if NA
                        pass
                    else: #else has data
                        if self.data[row][col_indexes[index2]] < min:
                            min = self.data[row][col_indexes[index2]] #finds a new min
                curr_stats.append(min) #adds the min to currstats
                max  = self.data[0][col_indexes[index2]] #gets a temporary max
                for row in range(len(self.data)):
                    if self.data[row][col_indexes[index2]] == "NA": #if equal to na skip
                        pass
                    else: #else is a valid value
                        if self.data[row][col_indexes[index2]] > max: #if its a new max
                            max = self.data[row][col_indexes[index2]] #set new max
                curr_stats.append(max) #add to stats
                mid = (min + max) / 2 #get the mid
                curr_stats.append(mid) #add middle to the to stats
                sum = 0
                count = 0
                for row in range(len(self.data)):
                    if self.data[row][col_indexes[index2]] == "NA": #if equal to na skip
                        pass
                    else:#else is a valid value
                        sum += self.data[row][col_indexes[index2]] #build up sum as traversing
                        count +=1 #get len of valid data
                average = sum / count #calculate average
                curr_stats.append(average) #append to stats
                sorted = []
                for row in range(len(self.data)):
                    if self.data[row][col_indexes[index2]] == "NA":#if equal to na skip
                        pass
                    else:#else is a valid value
                        sorted.append(self.data[row][col_indexes[index2]]) #add valid data to list
                sorted.sort() #sort the data
                if (len(sorted) % 2) == 0:
                    upper_index = int(len(sorted) / 2) #get
                    median_upper = sorted[upper_index]
                    lower_index = int(len(sorted) / 2 -1)
                    median_lower = sorted[lower_index]
                    median = (median_upper + median_lower) / 2 #calclate median
                    curr_stats.append(median) #add median to currstats
                else:
                    index = int((len(sorted) -1) / 2)
                    median = sorted[index] #grab median
                    curr_stats.append(median) #add meddian to stats
                curr_table.append(curr_stats) #add stats to table array
                index2 +=1 #increase index by 1
            names = ["attributes", "min", "max", "mid", "avg", "median"]
            new_table = MyPyTable(names, curr_table) #create new table
            return new_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        col_names = []
        data = []
        col_copy = copy.deepcopy(other_table.column_names)
        for title in key_column_names:
            col_copy.remove(title) #remove titles
        col_names = self.column_names + col_copy
        for row in self.data: #loop through row
            keys = []
            for key in key_column_names:
                index = self.column_names.index(key) #get index
                val = row[index]#get val
                keys.append(val) #append value
            for row2 in other_table.data:
                more_keys = []
                for key in key_column_names:
                    index = other_table.column_names.index(key) #get index
                    val = row2[index]#get value
                    more_keys.append(val) #add to more keys
                if keys == more_keys:
                    other_copy = copy.deepcopy(row2)
                    for val in more_keys: #removes values
                        other_copy.remove(val)
                    row_copy = row + other_copy #adds together arrays
                    data.append(row_copy)
        inner_join_table = MyPyTable(col_names, data) #creates a new table
        return inner_join_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        col_names = []
        data = []
        #gets a coyp fo the left and right side of the table
        left_table_col_copy = copy.deepcopy(self.column_names)
        right_table_col_copy = copy.deepcopy(other_table.column_names)
        left_table_data_copy = copy.deepcopy(self.data)
        for title in key_column_names:
            #removes the tiles from each
            right_table_col_copy.remove(title)
            left_table_col_copy.remove(title)
        col_names = self.column_names + right_table_col_copy #combines each array
        index_array = []
        for row in left_table_data_copy: #loop throw left side
            self_keys = []
            found = False
            for title in key_column_names:
                index = self.column_names.index(title) #get index
                val = row[index]#get val
                self_keys.append(val) #add to self key
            match_index = 0
            for other_row in other_table.data: #for other side
                other_keys = []
                for title in key_column_names:
                    index = other_table.column_names.index(title) #get index
                    val = other_row[index]#get the vaule
                    other_keys.append(val) #add to other keys
                if self_keys == other_keys: #if they are the same
                    index_array.append(match_index) #add to index array
                    found = True
                    other_copy = copy.deepcopy(other_row)
                    for key in other_keys:
                        other_copy.remove(key) #remove those keys
                    data_copy = row + other_copy #combine data together
                    data.append(data_copy)
                match_index = match_index + 1
            if not found:
                data_copy = row
                for _ in right_table_col_copy: #look at data that is not present
                    data_copy.append("NA") #add na if the not in
                data.append(data_copy)
        invalid_indexes = []
        valid_indexes = []
        for i in range(len(col_names)): #loop through coloumns
            if not col_names[i] in other_table.column_names:
                invalid_indexes.append(i) #add invalid indexes
            else:
                valid_indexes.append(other_table.column_names.index(col_names[i])) #add valid indexes
        #make a copy of the orignal data again
        other_data = copy.deepcopy(other_table.data)
        left_table_col_copy = copy.deepcopy(self.column_names)
        right_table_col_copy = copy.deepcopy(other_table.column_names)
        right_reorder = []
        for titles_left in left_table_col_copy: #loop throgh titles left
            for titles_right in right_table_col_copy: #loop through titles right
                if titles_left == titles_right:
                    right_reorder.append(right_table_col_copy.index(titles_left)) #get index of each
        for titles_right in right_table_col_copy: #loops throuogh right only
            if right_table_col_copy.index(titles_right) not in right_reorder: #check that not in right
                right_reorder.append(right_table_col_copy.index(titles_right)) #rearange the rights data
        other_data_copy = []
        for values in other_data: #for values in other data
            temp = []
            for indexes in right_reorder: #if index in rigth reordr
                temp.append(values[indexes]) #add values
            other_data_copy.append(temp) #add rearranged data
        other_data = other_data_copy #set other data to new rearragned data
        for row in other_data:
            for i in range(len(row)):
                if not i in valid_indexes:
                    del row[i] #delete invlaid data
        for row in other_data:
            for i in invalid_indexes:
                row.insert(i, "NA") #insiset NA where no data
        for i in range(len(other_data)):
            if not i in index_array:
                data.append(other_data[i]) #add good data
        outer_join_table = MyPyTable(col_names, data) #create new table
        return outer_join_table

if __name__ == "__main__":
    mpg_fname = os.path.join("input_data", "auto-mpg.txt")
    prices_fname = os.path.join("input_data", "auto-prices.txt")
    print(mpg_fname, prices_fname)
    print("\n----------------------------------\n")
    mpg_fname_table = MyPyTable().load_from_file(mpg_fname)
    prices_fname_table = MyPyTable().load_from_file(mpg_fname)
    print("MPG_FNAME", mpg_fname_table)
    print("PRICES_FNAME", prices_fname_table)
    print("MPG_FNAME columns", mpg_fname_table.column_names)
    print("PRICES_FNAME columns", prices_fname_table.column_names)
    print("\n----------------------------------\n")
    print("Shape of mpg_fname: ", mpg_fname_table.get_shape())
    print("Shape of mpg_fname: ", prices_fname_table.get_shape())
    print("\n----------------------------------\n")
    mpg_fname_car_name_cols = mpg_fname_table.get_column("car name", True)
    COUNT = 0
    print("Get column from mpg_fname: ")
    for car in mpg_fname_car_name_cols:
        print("\t", car)
        if COUNT == 10:
            break
        COUNT += 1
    #print("Get column  of mpg_fname: ", mpg_fname_car_name_cols)
    print("\n----------------------------------\n")
    prices_fname_car_name_cols = prices_fname_table.get_column("car name", True)
    COUNT = 0
    print("Get column from price_fname: ")
    for car in prices_fname_car_name_cols:
        print("\t", car)
        if COUNT == 10:
            break
        COUNT += 1
    #print("Get column  of mpg_fname: ", mpg_fname_car_name_cols)
    print("\n----------------------------------\n")
    prices_fname_car_name_cols_true = prices_fname_table.get_column("car name", True)
    mpg_fname_car_name_cols_true = mpg_fname_table.get_column("car name", True)
    prices_fname_car_name_cols_false = prices_fname_table.get_column("car name", True)
    mpg_fname_car_name_cols_false = mpg_fname_table.get_column("car name", True)
    print("mpg_frame get column with True vs mpg_frame get column with False")
    print("\t", len(mpg_fname_car_name_cols_true), len(mpg_fname_car_name_cols_false))
    print("prices_frame get column with True vs prices_frame get column with False")
    print("\t", len(prices_fname_car_name_cols_true), len(prices_fname_car_name_cols_false))
    print("\n----------------------------------\n")
    print("mpg_fname before and after removing missing values")
    before = mpg_fname_table.get_shape()
    mpg_fname_table.remove_rows_with_missing_values()
    print("\t", before, mpg_fname_table.get_shape())
    print("\n----------------------------------\n")
    sum_prices = prices_fname_table.compute_summary_statistics(["horsepower"])
    print("Summary Stats of Prices_Fname: ", sum_prices.data)
    sum_mpg = mpg_fname_table.compute_summary_statistics(["mpg", "weight"])
    for value in sum_mpg.data:
        print("Summary Stats of MPG_Fname: ", value)
    print("\n----------------------------------\n")
    copy_mpg_fname = copy.deepcopy(mpg_fname_table)
    copy_mpg_fname.drop_rows([0])
    print("mpg_frame dropping first column: \n\tOriginal")
    for values in mpg_fname_table.data[:5]:
        print("\t", values)
    print("\tDropped")
    for values in copy_mpg_fname.data[:5]:
        print("\t", values)
    print("\n----------------------------------\n")
    duplicates = mpg_fname_table.find_duplicates(["mpg"])
    print("mpg_frame checking for duplciates: ", )
    COUNT = 0
    for dup in duplicates:
        print("\t", mpg_fname_table.data[dup])
        COUNT += 1
        if COUNT == 10:
            break
    print("\n----------------------------------\n")
    join = ["car name", "model year"]
    mpg_price_table = mpg_fname_table.perform_full_outer_join(prices_fname_table, join)
    print("mpg and price tables full outer join: ")
    COUNT = 0
    for val in mpg_price_table.data:
        print("\t", val)
        COUNT += 1
        if COUNT == 10:
            break
    print("\n----------------------------------\n")
    mpg_fname_table.replace_missing_values_with_column_average("mpg")
    print("mpg tables replace missing values with column average: ")
    COUNT = 0
    for val in mpg_fname_table.data:
        print("\t", val)
        COUNT += 1
        if COUNT == 10:
            break
    print("\n----------------------------------\n")
    price_mpg_table = prices_fname_table.perform_inner_join(mpg_fname_table, join)
    print("mpg and price tables inner outer join: ")
    COUNT = 0
    for val in price_mpg_table.data:
        print("\t", val)
        COUNT += 1
        if COUNT == 10:
            break
    print("\n----------------------------------\n")
