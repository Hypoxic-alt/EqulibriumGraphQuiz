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
    Simulate the reaction in three phases:
      - Phase 1: t = 0 to 200 (base conditions)
      - Phase 2: t = 200 to 400 (after perturbation 1)
      - Phase 3: t = 400 to 600 (after perturbation 2)
      
    The perturbations adjust rate constants (temperature) or modify concentrations (volume/pressure or addition).
    For addition, the provided add_index indicates which species is affected.
    """
    # Base rate constants.
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
    a_coef, b_coef, c_coef, d_coef = scenario["a"], scenario["b"], scenario["c"], scenario["d"]

    # --- Phase 1 ---
    sol1 = odeint(generic_reaction, init, t1, args=(current_k1, current_k2, a_coef, b_coef, c_coef, d_coef))

    # --- Apply Perturbation 1 at t = 200 ---
    if "Temperature" in pert1:
        factor = 1 + temp_effect if "Increased" in pert1 else 1 - temp_effect
        # For exothermic (ΔH < 0) increasing temperature shifts equilibrium toward reactants (modify k2)
        # For endothermic (ΔH > 0) increasing temperature shifts equilibrium toward products (modify k1)
        if scenario["deltaH"] < 0:
            current_k2 *= factor
        else:
            current_k1 *= factor
        init2 = sol1[-1]
    elif "Volume" in pert1:
        if "Increased" in pert1:
            init2 = sol1[-1] / (1 + vol_effect)
        else:
            init2 = sol1[-1] * (1 + vol_effect)
    elif "Pressure" in pert1:
        if "Increased" in pert1:
            init2 = sol1[-1] * (1 + vol_effect)
        else:
            init2 = sol1[-1] / (1 + vol_effect)
    elif "Addition" in pert1:
        init2 = sol1[-1].copy()
        init2[add_index1] *= (1 + add_effect)
    else:
        init2 = sol1[-1]

    # --- Phase 2 ---
    sol2 = odeint(generic_reaction, init2, t2, args=(current_k1, current_k2, a_coef, b_coef, c_coef, d_coef))

    # --- Apply Perturbation 2 at t = 400 ---
    if "Temperature" in pert2:
        factor = 1 + temp_effect if "Increased" in pert2 else 1 - temp_effect
        if scenario["deltaH"] < 0:
            current_k2 *= factor
        else:
            current_k1 *= factor
        init3 = sol2[-1]
    elif "Volume" in pert2:
        if "Increased" in pert2:
            init3 = sol2[-1] / (1 + vol_effect)
        else:
            init3 = sol2[-1] * (1 + vol_effect)
    elif "Pressure" in pert2:
        if "Increased" in pert2:
            init3 = sol2[-1] * (1 + vol_effect)
        else:
            init3 = sol2[-1] / (1 + vol_effect)
    elif "Addition" in pert2:
        init3 = sol2[-1].copy()
        init3[add_index2] *= (1 + add_effect)
    else:
        init3 = sol2[-1]

    # --- Phase 3 ---
    sol3 = odeint(generic_reaction, init3, t3, args=(current_k1, current_k2, a_coef, b_coef, c_coef, d_coef))

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))
    # For each species, plot its time course using fixed colors and connect the phases.
    if a_coef != 0:
        ax.plot(t1, sol1[:, 0], color='blue')
        ax.plot(t2, sol2[:, 0], color='blue')
        ax.plot(t3, sol3[:, 0], color='blue')
        # Connect phase endpoints vertically:
        ax.plot([t1[-1], t1[-1]], [sol1[-1, 0], sol2[0, 0]], color='blue')
        ax.plot([t2[-1], t2[-1]], [sol2[-1, 0], sol3[0, 0]], color='blue')
    if b_coef != 0:
        ax.plot(t1, sol1[:, 1], color='red')
        ax.plot(t2, sol2[:, 1], color='red')
        ax.plot(t3, sol3[:, 1], color='red')
        ax.plot([t1[-1], t1[-1]], [sol1[-1, 1], sol2[0, 1]], color='red')
        ax.plot([t2[-1], t2[-1]], [sol2[-1, 1], sol3[0, 1]], color='red')
    if c_coef != 0:
        ax.plot(t1, sol1[:, 2], color='green')
        ax.plot(t2, sol2[:, 2], color='green')
        ax.plot(t3, sol3[:, 2], color='green')
        ax.plot([t1[-1], t1[-1]], [sol1[-1, 2], sol2[0, 2]], color='green')
        ax.plot([t2[-1], t2[-1]], [sol2[-1, 2], sol3[0, 2]], color='green')
    if d_coef != 0:
        ax.plot(t1, sol1[:, 3], color='purple')
        ax.plot(t2, sol2[:, 3], color='purple')
        ax.plot(t3, sol3[:, 3], color='purple')
        ax.plot([t1[-1], t1[-1]], [sol1[-1, 3], sol2[0, 3]], color='purple')
        ax.plot([t2[-1], t2[-1]], [sol2[-1, 3], sol3[0, 3]], color='purple')

    # Mark the perturbation times.
    ax.axvline(x=200, color="grey", linestyle="--")
    ax.axvline(x=400, color="grey", linestyle="--")
    ax.set_xlabel("Time")
    ax.set_ylabel("Concentration")
    ax.set_title(f"{scenario['name']}\n{scenario['reaction']}  |  ΔH = {scenario['deltaH']} kJ/mol")
    fig.tight_layout()
    return fig

# --------------------- Option Generator ---------------------
def generate_options(correct, scenario, vp_rep):
    """
    Build a pool of answer options and return 4 unique options (including the correct one)
    in randomized order. The pool is based on:
      - Temperature: "Increased Temperature", "Decreased Temperature"
      - Volume/Pressure: (based on vp_rep)
      - Addition: For reactants and products.
      
    If the reaction has more than one reactant (or product) but the stoichiometric coefficients
    are equal, only one addition option is provided for that category.
    """
    temp_opts = ["Increased Temperature", "Decreased Temperature"]
    vp_opts = ["Increased Volume", "Decreased Volume"] if vp_rep == "Volume" else ["Increased Pressure", "Decreased Pressure"]
    
    # Generate addition options for reactants.
    add_opts_reactants = []
    if scenario.get("reactants"):
        if len(scenario["reactants"]) > 1 and scenario["a"] == scenario["b"] and scenario["a"] != 0:
            # If both reactants are present in equal ratio, include only one option.
            add_opts_reactants.append(f"Addition of Reactant: {scenario['reactants'][0]}")
        else:
            for r in scenario["reactants"]:
                add_opts_reactants.append(f"Addition of Reactant: {r}")
    
    # Generate addition options for products.
    add_opts_products = []
    if scenario.get("products"):
        if len(scenario["products"]) > 1 and scenario["c"] == scenario["d"] and scenario["c"] != 0:
            add_opts_products.append(f"Addition of Product: {scenario['products'][0]}")
        else:
            for p in scenario["products"]:
                add_opts_products.append(f"Addition of Product: {p}")
    
    add_opts = add_opts_reactants + add_opts_products

    pool = temp_opts + vp_opts + add_opts
    if correct not in pool:
        pool.append(correct)
        
    # Ensure that if the correct answer is in a given category, its counterpart is included.
    if "Temperature" in correct:
        other_temp = [opt for opt in temp_opts if opt != correct]
        if other_temp and other_temp[0] not in pool:
            pool.append(other_temp[0])
    elif "Volume" in correct or "Pressure" in correct:
        other_vp = [opt for opt in vp_opts if opt != correct]
        if other_vp and other_vp[0] not in pool:
            pool.append(other_vp[0])
    elif "Addition" in correct:
        # For addition, if the reaction has equal ratio reactants/products, only one option is in add_opts.
        if "Reactant" in correct:
            others = [opt for opt in add_opts_reactants if opt != correct]
            if others:
                pool.append(others[0])
        elif "Product" in correct:
            others = [opt for opt in add_opts_products if opt != correct]
            if others:
                pool.append(others[0])
                
    pool = list(set(pool))
    if correct not in pool:
        pool.append(correct)
    # Finally, choose exactly 4 options (ensuring the correct one is present)
    if len(pool) > 4:
        # Force inclusion of the correct answer and one plausible confounder if available.
        forced = []
        if "Temperature" in correct:
            forced = [opt for opt in temp_opts if opt != correct]
        elif "Volume" in correct or "Pressure" in correct:
            forced = [opt for opt in vp_opts if opt != correct]
        elif "Addition" in correct:
            if "Reactant" in correct:
                forced = [opt for opt in add_opts_reactants if opt != correct]
            elif "Product" in correct:
                forced = [opt for opt in add_opts_products if opt != correct]
        forced = forced[:1]  # include only one forced option
        remaining = [opt for opt in pool if opt not in forced and opt != correct]
        count_needed = 4 - 1 - len(forced)
        if len(remaining) >= count_needed:
            chosen = random.sample(remaining, count_needed)
        else:
            chosen = remaining
        final_options = [correct] + forced + chosen
    else:
        final_options = pool

    while len(final_options) < 4:
        candidate = random.choice(temp_opts + vp_opts + add_opts)
        if candidate not in final_options:
            final_options.append(candidate)
    final_options = list(set(final_options))
    if correct not in final_options:
        final_options.append(correct)
    if len(final_options) > 4:
        if correct in final_options:
            final_options.remove(correct)
            extras = random.sample(final_options, 3)
            final_options = [correct] + extras
        else:
            final_options = random.sample(final_options, 4)
    random.shuffle(final_options)
    return final_options

# --------------------- Reaction Bank ---------------------
# Four original reactions plus two additional real-world examples.
reaction_bank = [
    {
        "name": "Haber Process",
        "reaction": "N₂ + 3H₂ ↔ 2NH₃",
        "a": 1, "b": 3, "c": 2, "d": 0,
        "deltaH": -92,
        "reactants": ["N₂", "H₂"],
        "products": ["NH₃"]
    },
    {
        "name": "Contact Process",
        "reaction": "2SO₂ + O₂ ↔ 2SO₃",
        "a": 2, "b": 1, "c": 2, "d": 0,
        "deltaH": -100,
        "reactants": ["SO₂", "O₂"],
        "products": ["SO₃"]
    },
    {
        "name": "Hydrogen Iodide Equilibrium",
        "reaction": "H₂ + I₂ ↔ 2HI",
        "a": 1, "b": 1, "c": 2, "d": 0,
        "deltaH": 52,
        "reactants": ["H₂", "I₂"],
        "products": ["HI"]
    },
    {
        "name": "Water-Gas Shift Reaction",
        "reaction": "CO + H₂O ↔ CO₂ + H₂",
        "a": 1, "b": 1, "c": 1, "d": 1,
        "deltaH": 20,
        "reactants": ["CO", "H₂O"],
        "products": ["CO₂", "H₂"]
    },
    {
        "name": "Carbon Dioxide Hydration",
        "reaction": "CO₂ + H₂O ↔ H₂CO₃",
        "a": 1, "b": 1, "c": 1, "d": 0,
        "deltaH": -20,
        "reactants": ["CO₂", "H₂O"],
        "products": ["H₂CO₃"]
    },
    {
        "name": "Nitrogen Dioxide Dimerization",
        "reaction": "2NO₂ ↔ N₂O₄",
        "a": 2, "b": 0, "c": 1, "d": 0,
        "deltaH": -57,
        "reactants": ["NO₂"],
        "products": ["N₂O₄"]
    }
]

# --------------------- Main Quiz App ---------------------
st.title("Reaction Equilibrium Quiz")

# Automatically generate the first question if not already done.
if "quiz_generated" not in st.session_state:
    scenario = random.choice(reaction_bank)
    st.session_state.scenario = scenario

    # Randomly choose a representation for volume/pressure.
    vp_representation = random.choice(["Volume", "Pressure"])
    st.session_state.vp_representation = vp_representation

    temp_opts = ["Increased Temperature", "Decreased Temperature"]
    vp_opts = ["Increased Volume", "Decreased Volume"] if vp_representation == "Volume" else ["Increased Pressure", "Decreased Pressure"]
    add_opts = ["Addition of Reactant", "Addition of Product"]
    perturbation_pool = temp_opts + vp_opts + add_opts

    chosen = random.sample(perturbation_pool, 2)
    pert1 = chosen[0]
    pert2 = chosen[1]
    add_index1 = None
    add_index2 = None

    # For addition options, choose a chemical from the appropriate list.
    if "Addition of Reactant" in pert1:
        available = []
        if scenario["a"] > 0:
            available.append(0)
        if scenario["b"] > 0:
            available.append(1)
        if available:
            add_index1 = random.choice(available)
            # Use the appropriate reactant; if both exist but are equal in ratio, only one option is used.
            if len(scenario["reactants"]) > 1 and scenario["a"] == scenario["b"]:
                pert1 = f"Addition of Reactant: {scenario['reactants'][0]}"
            else:
                pert1 = f"Addition of Reactant: {scenario['reactants'][available.index(add_index1)]}"
        else:
            pert1 = "Addition of Reactant: Unknown"
    elif "Addition of Product" in pert1:
        available = []
        if scenario["c"] > 0:
            available.append(2)
        if scenario["d"] > 0:
            available.append(3)
        if available:
            add_index1 = random.choice(available)
            if len(scenario["products"]) > 1 and scenario["c"] == scenario["d"]:
                pert1 = f"Addition of Product: {scenario['products'][0]}"
            else:
                pert1 = f"Addition of Product: {scenario['products'][available.index(add_index1)]}"
        else:
            pert1 = "Addition of Product: Unknown"

    if "Addition of Reactant" in pert2:
        available = []
        if scenario["a"] > 0:
            available.append(0)
        if scenario["b"] > 0:
            available.append(1)
        if available:
            add_index2 = random.choice(available)
            if len(scenario["reactants"]) > 1 and scenario["a"] == scenario["b"]:
                pert2 = f"Addition of Reactant: {scenario['reactants'][0]}"
            else:
                pert2 = f"Addition of Reactant: {scenario['reactants'][available.index(add_index2)]}"
        else:
            pert2 = "Addition of Reactant: Unknown"
    elif "Addition of Product" in pert2:
        available = []
        if scenario["c"] > 0:
            available.append(2)
        if scenario["d"] > 0:
            available.append(3)
        if available:
            add_index2 = random.choice(available)
            if len(scenario["products"]) > 1 and scenario["c"] == scenario["d"]:
                pert2 = f"Addition of Product: {scenario['products'][0]}"
            else:
                pert2 = f"Addition of Product: {scenario['products'][available.index(add_index2)]}"
        else:
            pert2 = "Addition of Product: Unknown"

    st.session_state.perturbation1 = pert1
    st.session_state.perturbation2 = pert2
    st.session_state.add_index1 = add_index1
    st.session_state.add_index2 = add_index2

    # Generate and store the answer options.
    st.session_state.opts_q1 = generate_options(pert1, scenario, vp_representation)
    st.session_state.opts_q2 = generate_options(pert2, scenario, vp_representation)
    st.session_state.submitted = False
    st.session_state.quiz_generated = True

# --- Quiz Display ---
fig = simulate_quiz_reaction(
    st.session_state.scenario,
    st.session_state.perturbation1,
    st.session_state.perturbation2,
    add_index1=st.session_state.add_index1,
    add_index2=st.session_state.add_index2
)
st.pyplot(fig)

st.write("Two perturbations were applied at **t = 200** and **t = 400**. Based on the plot, select the perturbation that occurred at each time:")

opts_q1 = st.session_state.opts_q1
opts_q2 = st.session_state.opts_q2

user_answer1 = st.radio("Perturbation at **t = 200**:", opts_q1, key="ans1")
user_answer2 = st.radio("Perturbation at **t = 400**:", opts_q2, key="ans2")

if st.button("Submit Answers"):
    correct1 = st.session_state.perturbation1
    correct2 = st.session_state.perturbation2
    st.session_state.submitted = True
    if user_answer1 == correct1 and user_answer2 == correct2:
        st.success("Correct! Well done.")
    else:
        st.error(f"Incorrect.\n\n**Correct Answers:**\n- t = 200: {correct1}\n- t = 400: {correct2}")

# Show the Next Question button after answers have been submitted.
if st.session_state.get("submitted", False):
    if st.button("Next Question"):
        # Clear all session state variables to generate a new question.
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
