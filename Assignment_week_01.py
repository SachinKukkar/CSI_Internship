# Triangle Pattern Program with User Input
# This program asks the user to enter the number of rows
# and then prints a left-aligned triangle pattern using asterisks (*)

# Step 1: Take input from the user for number of rows
# input() function returns a string, so we convert it to integer using int()

num_rows = int(input("Enter the number of rows for the triangle pattern: "))

# Step 2: Loop through each row from 1 to num_rows (inclusive)
for row in range(1, num_rows + 1):
    # Step 3: Print 'row' number of asterisks with a space
    for col in range(row):
        print("*", end=" ")  # end=" " keeps the output on the same line with a space
    # Step 4: After each row, print a newline to move to the next line
    print()
