# Simple Reflex Agent Simulation

## 🧠 Overview
This Python script simulates a **Simple Reflex Agent** operating in a **2x2 grid environment**. The agent follows a basic rule-based logic: it cleans rooms that are dirty and moves sequentially through the environment when rooms are clean.

---

## ✨ Features

- ✅ 2x2 Grid Environment with 4 Rooms (Room1 to Room4)
- 📊 Visual representation using `matplotlib`
- 🧽 Simple reflex agent that:
  - Cleans dirty rooms
  - Moves to the next room if current room is clean
- ⏱️ Step-by-step visualization of agent actions (8 simulation steps)

---

## 🖥️ How to Run

### 1. Prerequisites
Make sure Python 3 is installed on your system.

### 2. Install Dependencies

```bash
pip install matplotlib
```

### 3. Run the Simulation

```bash
python simple_reflex_agent.py
```

---

## 🔍 Expected Output

The simulation will display 8 steps in which the agent performs the following actions:

1. Starts in **Room1** (clean) and moves to **Room2**
2. Finds **Room2** dirty and cleans it
3. Moves to **Room3**
4. Cleans if dirty
5. Moves to **Room4**
6. Cleans if dirty
7. Moves back to **Room1**
8. The cycle continues...

Each step is visualized in a plot showing:

- The agent's position  
- Room cleanliness status  
- Agent's current action (move or clean)

---

## 📂 File Structure

```bash
.
├── simple_reflex_agent.py   # Main simulation script
└── README.md                # This file
```

---

## 📌 Notes

- You can customize the initial room states (dirty/clean) inside `simple_reflex_agent.py`.
- The simulation is simplified for educational/demo purposes.
