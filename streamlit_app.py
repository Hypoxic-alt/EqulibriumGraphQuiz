import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random

# --- Differential Equation Model ---
def generic_reaction(concentrations, t, k1, k2, a, b, c, d):
    """ODE for a generic reversible reaction: aA + bB <-> cC + dD."""
    A, B, C, D = concentrations
    # Convention: X^0 = 1.
    r_forward = k1 * (A ** a) * (B ** b)
    r_reverse = k2 * (C ** c) * (D ** d)
    r = r_forward - r_reverse
    dA_dt = -a * r
    dB_dt = -b * r
    dC_dt =  c * r
    dD_dt =  d * r
    return [dA_dt, dB_dt, dC_dt, dD_dt]

# --- Simulation Function for the Quiz ---
def simulate_quiz_reaction(scenario, pert1, pert2, 
                           temp_effect=0.2, vol_effect=0.2, add_effect=0.2):
    """
    Simulate the reaction in three phases:
      - Phase 1 (t=0 to 200): base conditions.
      - At t=200, apply perturbation 1.
      - Phase 2 (t=200 to 400): conditions after perturbation 1.
      - At t=400, apply perturbation 2.
      - Phase 3 (t=400 to 600): conditions after perturbation 2.
    """
    # Unpack scenario parameters
    a = scenario["a"]
    b = scenario["b"]
    c = scenario["c"]
    d = scenario["d"]
    reaction_type = scenario["reaction_type"]
    
    # Base rate constants
    k1_base = 0.02
    k2_base = 0.01
    current_k1 = k1_base
    current_k2 = k2_base
    
    # Define time arrays for the three phases
    t1 = np.linspace(0, 200, 500)
    t2 = np.linspace(200, 400, 500)
    t3 = np.linspace(400, 600, 500)
    
    # Initial conditions: assume starting with 1.0 for A and B; 0 for products.
    init = np.array([1.0, 1.0, 0.0, 0.0])
    
    # --- Phase 1 ---
    sol1 = odeint(generic_reaction, init, t1, args=(current_k1, current_k2, a, b, c, d))
    
    # --- Apply Perturbation 1 at t=200 ---
    if pert1 == "Temperature":
        # For a temperature change, update the appropriate rate constant.
        if reaction_type == "Exothermic":
            # Exothermic: increase in temperature shifts equilibrium toward reactants
            # by increasing k2.
            current_k2 = current_k2 * (1 + temp_effect)
        else:  # Endothermic
            current_k1 = current_k1 * (1 + temp_effect)
        init2 = sol1[-1]
    elif pert1 == "Volume":
        # Increase in volume lowers concentrations.
        init2 = sol1[-1] / (1 + vol_effect)
    elif pert1 == "Addition":
        # Addition of a reactant (we choose species A).
        init2 = sol1[-1].copy()
        init2[0] = init2[0] * (1 + add_effect)
    
    # --- Phase 2 ---
    sol2 = odeint(generic_reaction, init2, t2, args=(current_k1, current_k2, a, b, c, d))
    
    # --- Apply Perturbation 2 at t=400 ---
    if pert2 == "Temperature":
        if reaction_type == "Exothermic":
            current_k2 = current_k2 * (1 + temp_effect)
        else:
            current_k1 = current_k1 * (1 + temp_effect)
        init3 = sol2[-1]
    elif pert2 == "Volume":
        init3 = sol2[-1] / (1 + vol_effect)
    elif pert2 == "Addition":
        init3 = sol2[-1].copy()
        init3[0] = init3[0] * (1 + add_effect)
    
    # --- Phase 3 ---
    sol3 = odeint(generic_reaction, init3, t3, args=(current_k1, current_k2, a, b, c, d))
    
    # Combine time and solutions for plotting
    t_total = np.concatenate([t1, t2[1:], t3[1:]])
    A_total = np.concatenate([sol1[:, 0], sol2[1:, 0], sol3[1:, 0]])
    B_total = np.concatenate([sol1[:, 1], sol2[1:, 1], sol3[1:, 1]])
    C_total = np.concatenate([sol1[:, 2], sol2[1:, 2], sol3[1:, 2]])
    D_total = np.concatenate([sol1[:, 3], sol2[1:, 3], sol3[1:, 3]])
    
    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each species if its stoichiometric coefficient is nonzero.
    if a != 0:
        ax.plot(t1, sol1[:, 0], label="A Phase 1", color="blue")
        ax.plot(t2, sol2[:, 0], label="A Phase 2", color="blue")
        ax.plot(t3, sol3[:, 0], label="A Phase 3", color="blue")
    if b != 0:
        ax.plot(t1, sol1[:, 1], label="B Phase 1", color="red")
        ax.plot(t2, sol2[:, 1], label="B Phase 2", color="red")
        ax.plot(t3, sol3[:, 1], label="B Phase 3", color="red")
    if c != 0:
        ax.plot(t1, sol1[:, 2], label="C Phase 1", color="green")
        ax.plot(t2, sol2[:, 2], label="C Phase 2", color="green")
        ax.plot(t3, sol3[:, 2], label="C Phase 3", color="green")
    if d != 0:
        ax.plot(t1, sol1[:, 3], label="D Phase 1", color="purple")
        ax.plot(t2, sol2[:, 3], label="D Phase 2", color="purple")
        ax.plot(t3, sol3[:, 3], label="D Phase 3", color="purple")
    
    # Mark the perturbation times with vertical dashed lines.
    ax.axvline(x=200, color="grey", linestyle="--")
    ax.axvline(x=400, color="grey", linestyle="--")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    title_str = f"{scenario['name']}\n{scenario['reaction']}  |  Reaction Type: {reaction_type}"
    ax.set_title(title_str)
    ax.legend(fontsize=8)
    fig.tight_layout()
    
    return fig

# --- Reaction Bank ---
reaction_bank = [
    {
        "name": "Haber Process",
        "reaction": "N₂ + 3H₂ ↔ 2NH₃",
        "a": 1, "b": 3, "c": 2, "d": 0,
        "reaction_type": "Exothermic"
    },
    {
        "name": "Contact Process",
        "reaction": "2SO₂ + O₂ ↔ 2SO₃",
        "a": 2, "b": 1, "c": 2, "d": 0,
        "reaction_type": "Exothermic"
    },
    {
        "name": "Hydrogen Iodide Equilibrium",
        "reaction": "H₂ + I₂ ↔ 2HI",
        "a": 1, "b": 1, "c": 2, "d": 0,
        "reaction_type": "Endothermic"
    },
    {
        "name": "Water-Gas Shift Reaction",
        "reaction": "CO + H₂O ↔ CO₂ + H₂",
        "a": 1, "b": 1, "c": 1, "d": 1,
        "reaction_type": "Endothermic"
    }
]

# --- Mapping for Display ---
pert_display = {
    "Temperature": "Temperature change",
    "Volume": "Volume change",
    "Addition": "Addition of reactant/product"
}

# --- Main Quiz App ---
st.title("Reaction Equilibrium Quiz")

# Use session state to hold quiz data.
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False
if "show_result" not in st.session_state:
    st.session_state.show_result = False

# Start (or restart) the quiz.
if not st.session_state.quiz_started:
    if st.button("Start Quiz"):
        # Randomly choose one reaction scenario.
        scenario = random.choice(reaction_bank)
        st.session_state.scenario = scenario
        # Randomly choose two distinct perturbations.
        perturbations = random.sample(["Temperature", "Volume", "Addition"], 2)
        st.session_state.perturbation1 = perturbations[0]
        st.session_state.perturbation2 = perturbations[1]
        st.session_state.quiz_started = True
        st.session_state.show_result = False
else:
    st.subheader("Scenario")
    scenario = st.session_state.scenario
    st.markdown(f"**Reaction:** {scenario['reaction']}")
    st.markdown(f"**Reaction Name:** {scenario['name']}")
    st.markdown(f"**Reaction Type:** {scenario['reaction_type']}")
    
    # Run the simulation and display the plot.
    fig = simulate_quiz_reaction(scenario, st.session_state.perturbation1, st.session_state.perturbation2)
    st.pyplot(fig)
    
    st.write("A reaction was simulated with two perturbations occurring at **t = 200** and **t = 400**. "
             "Based on the plot, **identify the perturbation that occurred at each time.**")
    
    # Quiz questions: use select boxes for the two time points.
    options = ["Temperature change", "Volume change", "Addition of reactant/product"]
    user_answer1 = st.selectbox("Select the perturbation at **t = 200**:", options, key="ans1")
    user_answer2 = st.selectbox("Select the perturbation at **t = 400**:", options, key="ans2")
    
    if st.button("Submit Answers"):
        correct1 = pert_display[st.session_state.perturbation1]
        correct2 = pert_display[st.session_state.perturbation2]
        if user_answer1 == correct1 and user_answer2 == correct2:
            st.success("Correct! Well done.")
        else:
            st.error(f"Incorrect. The correct answers were:\n- t = 200: {correct1}\n- t = 400: {correct2}")
        st.session_state.show_result = True
    
    if st.button("Reset Quiz"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()
