import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random

# --------------------- Reaction Model ---------------------
def generic_reaction(concentrations, t, k1, k2, a, b, c, d):
    """ODE for a generic reversible reaction: aA + bB ↔ cC + dD."""
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

# --------------------- Simulation Function ---------------------
def simulate_quiz_reaction(scenario, pert1, pert2, add_index1=None, add_index2=None,
                           temp_effect=0.2, vol_effect=0.2, add_effect=0.2):
    """
    Simulate a reaction in three phases:
      - Phase 1: t = 0 to 200 (base conditions)
      - Phase 2: t = 200 to 400 (after perturbation 1)
      - Phase 3: t = 400 to 600 (after perturbation 2)
      
    Depending on the detailed perturbation string, update rate constants (temperature changes)
    or modify concentrations (volume/pressure or addition changes). For addition, the provided
    add_index indicates which species is affected.
    """
    # Base rate constants
    k1_base = 0.02
    k2_base = 0.01
    current_k1 = k1_base
    current_k2 = k2_base

    # Time arrays for the three phases.
    t1 = np.linspace(0, 200, 500)
    t2 = np.linspace(200, 400, 500)
    t3 = np.linspace(400, 600, 500)

    # Initial conditions: assume [A]=1.0, [B]=1.0, [C]=0, [D]=0.
    init = np.array([1.0, 1.0, 0.0, 0.0])
    a, b, c, d = scenario["a"], scenario["b"], scenario["c"], scenario["d"]

    # --- Phase 1 ---
    sol1 = odeint(generic_reaction, init, t1, args=(current_k1, current_k2, a, b, c, d))

    # --- Apply Perturbation 1 at t = 200 ---
    if "Temperature" in pert1:
        if "Increased" in pert1:
            factor = 1 + temp_effect
        else:
            factor = 1 - temp_effect
        # For exothermic reactions, temperature change affects k2; for endothermic, k1.
        if scenario["reaction_type"] == "Exothermic":
            current_k2 *= factor
        else:
            current_k1 *= factor
        init2 = sol1[-1]
    elif ("Volume" in pert1) or ("Pressure" in pert1):
        # For volume/pressure: (increased volume or decreased pressure lowers concentrations)
        if pert1 in ["Increased Volume", "Decreased Pressure"]:
            init2 = sol1[-1] / (1 + vol_effect)
        elif pert1 in ["Decreased Volume", "Increased Pressure"]:
            init2 = sol1[-1] * (1 + vol_effect)
    elif "Addition" in pert1:
        init2 = sol1[-1].copy()
        # Increase the concentration of the chosen species by a factor (1 + add_effect).
        init2[add_index1] *= (1 + add_effect)
    else:
        init2 = sol1[-1]

    # --- Phase 2 ---
    sol2 = odeint(generic_reaction, init2, t2, args=(current_k1, current_k2, a, b, c, d))

    # --- Apply Perturbation 2 at t = 400 ---
    if "Temperature" in pert2:
        if "Increased" in pert2:
            factor = 1 + temp_effect
        else:
            factor = 1 - temp_effect
        if scenario["reaction_type"] == "Exothermic":
            current_k2 *= factor
        else:
            current_k1 *= factor
        init3 = sol2[-1]
    elif ("Volume" in pert2) or ("Pressure" in pert2):
        if pert2 in ["Increased Volume", "Decreased Pressure"]:
            init3 = sol2[-1] / (1 + vol_effect)
        elif pert2 in ["Decreased Volume", "Increased Pressure"]:
            init3 = sol2[-1] * (1 + vol_effect)
    elif "Addition" in pert2:
        init3 = sol2[-1].copy()
        init3[add_index2] *= (1 + add_effect)
    else:
        init3 = sol2[-1]

    # --- Phase 3 ---
    sol3 = odeint(generic_reaction, init3, t3, args=(current_k1, current_k2, a, b, c, d))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot each species if its coefficient is nonzero.
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

    # Mark the perturbation times.
    ax.axvline(x=200, color="grey", linestyle="--")
    ax.axvline(x=400, color="grey", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(f"{scenario['name']}\n{scenario['reaction']}  |  Reaction Type: {scenario['reaction_type']}")
    ax.legend(fontsize=8)
    fig.tight_layout()

    return fig

# --------------------- Reaction Bank ---------------------
# Each reaction now includes a reaction name, reaction string, stoichiometric coefficients,
# reaction type, and lists of reactants and products.
reaction_bank = [
    {
        "name": "Haber Process",
        "reaction": "N₂ + 3H₂ ↔ 2NH₃",
        "a": 1, "b": 3, "c": 2, "d": 0,
        "reaction_type": "Exothermic",
        "reactants": ["N₂", "H₂"],
        "products": ["NH₃"]
    },
    {
        "name": "Contact Process",
        "reaction": "2SO₂ + O₂ ↔ 2SO₃",
        "a": 2, "b": 1, "c": 2, "d": 0,
        "reaction_type": "Exothermic",
        "reactants": ["SO₂", "O₂"],
        "products": ["SO₃"]
    },
    {
        "name": "Hydrogen Iodide Equilibrium",
        "reaction": "H₂ + I₂ ↔ 2HI",
        "a": 1, "b": 1, "c": 2, "d": 0,
        "reaction_type": "Endothermic",
        "reactants": ["H₂", "I₂"],
        "products": ["HI"]
    },
    {
        "name": "Water-Gas Shift Reaction",
        "reaction": "CO + H₂O ↔ CO₂ + H₂",
        "a": 1, "b": 1, "c": 1, "d": 1,
        "reaction_type": "Endothermic",
        "reactants": ["CO", "H₂O"],
        "products": ["CO₂", "H₂"]
    }
]

# --------------------- Main Quiz App ---------------------
st.title("Reaction Equilibrium Quiz")

# Use session state to track quiz progress.
if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False

# --- Start/Restart Quiz ---
if not st.session_state.quiz_started:
    if st.button("Start Quiz"):
        # Randomly choose one reaction scenario.
        scenario = random.choice(reaction_bank)
        st.session_state.scenario = scenario

        # Create the full list of detailed perturbation options.
        # Temperature options:
        temp_opts = ["Increased Temperature", "Decreased Temperature"]
        # Volume/pressure options:
        vol_opts = ["Increased Volume", "Decreased Volume", "Increased Pressure", "Decreased Pressure"]
        # Addition options (base labels; we will later append the chosen chemical name):
        add_opts = ["Addition of Reactant", "Addition of Product"]
        perturbation_pool = temp_opts + vol_opts + add_opts

        # Randomly choose two distinct perturbations.
        chosen = random.sample(perturbation_pool, 2)
        # Initialize addition indices (if needed)
        add_index1 = None
        add_index2 = None
        pert1 = chosen[0]
        pert2 = chosen[1]

        # For addition options, choose a chemical from the appropriate list.
        # Reactants: species A (index 0) if available and species B (index 1) if coefficient > 0.
        if "Addition of Reactant" in pert1:
            available = []
            if scenario["a"] > 0:
                available.append(0)
            if scenario["b"] > 0:
                available.append(1)
            add_index1 = random.choice(available)
            # Map index to chemical name (assume order in scenario["reactants"])
            chem = scenario["reactants"][available.index(add_index1)] if len(scenario["reactants"]) > 1 else scenario["reactants"][0]
            pert1 = f"Addition of Reactant: {chem}"
        elif "Addition of Product" in pert1:
            available = []
            if scenario["c"] > 0:
                available.append(2)
            if scenario["d"] > 0:
                available.append(3)
            add_index1 = random.choice(available)
            # For products, species C corresponds to the first product, D to the second.
            if add_index1 == 2:
                chem = scenario["products"][0]
            else:
                chem = scenario["products"][1] if len(scenario["products"]) > 1 else scenario["products"][0]
            pert1 = f"Addition of Product: {chem}"

        if "Addition of Reactant" in pert2:
            available = []
            if scenario["a"] > 0:
                available.append(0)
            if scenario["b"] > 0:
                available.append(1)
            add_index2 = random.choice(available)
            chem = scenario["reactants"][available.index(add_index2)] if len(scenario["reactants"]) > 1 else scenario["reactants"][0]
            pert2 = f"Addition of Reactant: {chem}"
        elif "Addition of Product" in pert2:
            available = []
            if scenario["c"] > 0:
                available.append(2)
            if scenario["d"] > 0:
                available.append(3)
            add_index2 = random.choice(available)
            if add_index2 == 2:
                chem = scenario["products"][0]
            else:
                chem = scenario["products"][1] if len(scenario["products"]) > 1 else scenario["products"][0]
            pert2 = f"Addition of Product: {chem}"

        # Store the detailed perturbation strings and indices.
        st.session_state.perturbation1 = pert1
        st.session_state.perturbation2 = pert2
        st.session_state.add_index1 = add_index1
        st.session_state.add_index2 = add_index2
        st.session_state.quiz_started = True

# --- Quiz Display ---
if st.session_state.quiz_started:
    scenario = st.session_state.scenario
    st.subheader("Scenario Details")
    st.markdown(f"**Reaction:** {scenario['reaction']}")
    st.markdown(f"**Reaction Name:** {scenario['name']}")
    st.markdown(f"**Reaction Type:** {scenario['reaction_type']}")

    # Run the simulation and display the plot.
    fig = simulate_quiz_reaction(
        scenario,
        st.session_state.perturbation1,
        st.session_state.perturbation2,
        add_index1=st.session_state.add_index1,
        add_index2=st.session_state.add_index2
    )
    st.pyplot(fig)

    st.write("Two perturbations were applied at **t = 200** and **t = 400**. Based on the plot, **select the perturbation that occurred at each time.**")

    # Build a list of answer options.
    # Include the detailed temperature and volume/pressure options.
    options = [
        "Increased Temperature",
        "Decreased Temperature",
        "Increased Volume",
        "Decreased Volume",
        "Increased Pressure",
        "Decreased Pressure"
    ]
    # Also add all possible addition answers for reactants/products from this reaction.
    if scenario["a"] > 0 or scenario["b"] > 0:
        for chem in scenario["reactants"]:
            options.append(f"Addition of Reactant: {chem}")
    if scenario["c"] > 0 or scenario["d"] > 0:
        for chem in scenario["products"]:
            options.append(f"Addition of Product: {chem}")
    # Remove duplicates if any.
    options = list(dict.fromkeys(options))

    user_answer1 = st.selectbox("Select the perturbation at **t = 200**:", options, key="ans1")
    user_answer2 = st.selectbox("Select the perturbation at **t = 400**:", options, key="ans2")

    if st.button("Submit Answers"):
        correct1 = st.session_state.perturbation1
        correct2 = st.session_state.perturbation2
        if user_answer1 == correct1 and user_answer2 == correct2:
            st.success("Correct! Well done.")
        else:
            st.error(f"Incorrect.\n\n**Correct Answers:**\n- t = 200: {correct1}\n- t = 400: {correct2}")

    if st.button("Reset Quiz"):
        # Clear session state and restart the app.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
