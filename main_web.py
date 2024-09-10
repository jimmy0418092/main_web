# streamlit run main_web.py --server.address 0.0.0.0
# go to localhost:8501 on the computer or host_ip:8501 on other device

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import streamlit as st

def get_matrix_and_state(code, day_avgs):
    st.write(f"股票代碼: {code}")
    
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
    formatted_matrix = matrix.applymap(lambda x: f'{x:.2f}')
    return formatted_matrix

def calculate(code, day_avgs):
    try:
        transition_matrix, steady_state_probs, return_days = get_matrix_and_state(code, day_avgs)
        
        # Calculate the last state
        last_change = (day_avgs[-1] - day_avgs[-2]) / day_avgs[-2] * 100
        
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

        st.write("### Transition Matrix:")
        st.dataframe(format_matrix(transition_matrix))

        st.write("### Steady State Probabilities:")
        st.write(steady_state_probs.to_frame(name="Probability").style.format("{:.2f}"))

        st.write("### Return Days:")
        st.write(return_days.to_frame(name="Days").style.format("{:.2f}"))

        st.write(f"Last State: {last_state} (Change: {last_change:.2f}%)")
        st.write(f"Predicted Next State: {predicted_next_state}")
    except ValueError:
        st.error("請填寫正確的均價")
    except FileNotFoundError:
        st.error("找不到此股票的資料")

# Streamlit Interface
st.title("Stock Prediction")

# Input for stock code
code = st.text_input("Enter Stock Code:")

# Input for the last 5 days' average prices
day_avgs = []
day_avgs.append(st.number_input("Day 1:", value=0.0))
day_avgs.append(st.number_input("Day 2:", value=0.0))
day_avgs.append(st.number_input("Day 3:", value=0.0))
day_avgs.append(st.number_input("Day 4:", value=0.0))
day_avgs.append(st.number_input("Day 5:", value=0.0))

# Button to trigger the calculation
if st.button("Predict"):
    calculate(code, day_avgs)
