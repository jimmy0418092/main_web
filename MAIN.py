import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox

def get_matrix_and_state(code, day_avgs):
    print("股票代碼", code)
    
    # Assuming you already have the file and the structure
    df = pd.read_csv(os.path.join('data', f'{code}.csv'))

    # Calculate the day average price
    df['Day_Avg'] = (df['open'] + df['close']) / 2
    df['Pct_Change'] = df['Day_Avg'].pct_change() * 100

    # Fill NaN values in 'Pct_Change' with a moving average
    df['Pct_Change'].fillna(df['Pct_Change'].rolling(window=3, min_periods=1).mean(), inplace=True)

    # Define states based on the percentage change
    def categorize_state(change):
        if change > 10:
            return '++'
        elif change > 5:
            return '+'
        elif change > -5 and change <= 5:
            return '0'
        elif change > -10:
            return '-'
        else:
            return '--'

    df['State'] = df['Pct_Change'].apply(categorize_state)

    # Get the unique states in the defined order
    states = ['++', '+', '0', '-', '--']

    # Initialize the transition matrix with zeros
    transition_matrix = pd.DataFrame(0, index=states, columns=states)

    # Populate the transition matrix
    for (i, current_state), (_, next_state) in zip(df['State'].shift().items(), df['State'].items()):
        if pd.notna(current_state) and pd.notna(next_state):
            transition_matrix.at[current_state, next_state] += 1

    # Replace NaN with 0 (for cases with no transitions)
    transition_matrix.fillna(0, inplace=True)

    # Recalculate probabilities: Normalize each row if it sums to more than 0
    transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

    # Convert the transition matrix to a numpy array
    P = transition_matrix.values

    # Solve for steady state probabilities
    evals, evecs = np.linalg.eig(P.T)
    steady_state = np.real(evecs[:, np.isclose(evals, 1)])

    # Normalize the steady-state vector
    steady_state = steady_state / steady_state.sum()
    steady_state = steady_state.flatten()

    # Map the steady-state probabilities to the states
    steady_state_probs = pd.Series(steady_state, index=states)

    return transition_matrix, steady_state_probs, 1/steady_state_probs

def format_matrix(matrix):
    # Convert all float numbers to 2 decimal places and prepare for printing
    result = []
    for index, row in matrix.iterrows():
        formatted_row = [f'{index}'] + [f'{value:.2f}' for value in row]
        result.append(" | ".join(formatted_row))
    header = '  ' + " | ".join([""] + list(matrix.columns))
    return "\n".join([header] + result)

def calculate():
    try:
        code = entry_code.get()
        day_avgs = [
            float(entry1.get()),
            float(entry2.get()),
            float(entry3.get()),
            float(entry4.get()),
            float(entry5.get())
        ]
        transition_matrix, steady_state_probs, return_days = get_matrix_and_state(code, day_avgs)
        
        # Calculate the last state
        last_change = (float(entry5.get()) - float(entry4.get())) / float(entry4.get()) * 100
        
        def categorize_state(change):
            if change > 10:
                return '++'
            elif change > 5:
                return '+'
            elif change > -5 and change <= 5:
                return '0'
            elif change > -10:
                return '-'
            else:
                return '--'
        
        last_state = categorize_state(last_change)

        # Predict the next state using the transition matrix
        next_state_probabilities = transition_matrix.loc[last_state]
        predicted_next_state = next_state_probabilities.idxmax()

        result = f"Transition Matrix:\n{format_matrix(transition_matrix)}\n\n"
        result += f"Steady State Probabilities:\n{steady_state_probs.to_string(float_format='%.2f')}\n\n"
        result += f"Return days:\n{return_days.to_string(float_format='%.2f')}\n\n"
        result += f"Last State: {last_state} (Change: {last_change:.2f}%)\n"
        result += f"Predicted Next State: {predicted_next_state}"

        messagebox.showinfo("Prediction Result", result)
    except ValueError:
        messagebox.showerror("Input Error", "請填寫正確的均價")
    except FileNotFoundError:
        messagebox.showerror("Input Error", "找不到此股票的資料")

# Create the main window
root = tk.Tk()
root.title("Stock Prediction")

# Create and place labels and entries for stock code input
tk.Label(root, text="Enter Stock Code:").grid(row=0, column=0)
entry_code = tk.Entry(root)
entry_code.grid(row=0, column=1)

# Create and place labels and entries for input
tk.Label(root, text="Enter the last 5 days' average prices:").grid(row=1, columnspan=2)

tk.Label(root, text="Day 1:").grid(row=2, column=0)
entry1 = tk.Entry(root)
entry1.grid(row=2, column=1)

tk.Label(root, text="Day 2:").grid(row=3, column=0)
entry2 = tk.Entry(root)
entry2.grid(row=3, column=1)

tk.Label(root, text="Day 3:").grid(row=4, column=0)
entry3 = tk.Entry(root)
entry3.grid(row=4, column=1)

tk.Label(root, text="Day 4:").grid(row=5, column=0)
entry4 = tk.Entry(root)
entry4.grid(row=5, column=1)

tk.Label(root, text="Day 5:").grid(row=6, column=0)
entry5 = tk.Entry(root)
entry5.grid(row=6, column=1)

# Create and place the calculate button
calculate_button = tk.Button(root, text="Predict", command=calculate)
calculate_button.grid(row=7, columnspan=2)

# Run the application
root.mainloop()
