Simple Reflex Agent Simulation
Overview
This Python script demonstrates a simple reflex agent operating in a 2x2 grid environment. The agent cleans dirty rooms and moves between rooms based on a basic reflex rule.

Features
2x2 grid environment with 4 rooms

Visual representation using matplotlib

Simple reflex agent that:

Cleans dirty rooms

Moves to next room if current room is clean

Step-by-step visualization of agent actions

How to Run
Ensure you have Python installed

Install required packages:

bash
pip install matplotlib
Run the script:

bash
python simple_reflex_agent.py
Expected Output
The simulation will show 8 steps of the agent's operation:

Agent starts in Room1 (clean) and moves to Room2

Agent finds Room2 dirty and cleans it

Agent moves to Room3

Agent moves to Room4

Agent moves back to Room1

The cycle continues until all steps are completed