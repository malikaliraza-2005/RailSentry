# RailSentry: App Guide

This guide covers every page, button and option on the RailSentry website. You do not need any technical background. For installation, see the [README](../README.md).

Example numbers come from the results saved with the project. Yours may change after training again.

## Contents

1. [Open the app](#1-open-the-app)
2. [Five-minute tour](#2-five-minute-tour)
3. [The top of every page](#3-the-top-of-every-page)
4. [How to read the station map](#4-how-to-read-the-station-map)
5. [Playback controls](#5-playback-controls)
6. [Dashboard](#6-dashboard)
7. [Agent & Search](#7-agent--search)
8. [Data & ML](#8-data--ml)
9. [Neural Networks](#9-neural-networks)
10. [Smart System](#10-smart-system)
11. [Try Your Own Data](#11-try-your-own-data)
12. [Logs](#12-logs)
13. [Which buttons change saved results?](#13-which-buttons-change-saved-results)
14. [Messages and what to do](#14-messages-and-what-to-do)
15. [Questions](#15-questions)

---

## 1. Open the app

1. Double-click `start_ui.bat`.
2. Two black windows open. **Leave them open.**
3. Your browser opens <http://127.0.0.1:5173>.

If the page opens before everything is ready, wait a few seconds and refresh. To stop the app, close both windows.

---

## 2. Five-minute tour

Nothing here changes any saved results.

| Step | Page | Do this | Notice |
|:-:|---|---|---|
| 1 | Dashboard | Read the four large numbers | A summary of the project |
| 2 | Agent & Search | Watch the three maps | A\* finds the cheapest route |
| 3 | Agent & Search | **Agent playback**, then Scenario 5 | The vehicle meets hidden walls and re-plans |
| 4 | Data & ML | Click the location map | The point joins an area |
| 5 | Neural Networks | **Predict risk** | A risk level and a response time |
| 6 | Smart System | Compare the two routes | One avoids danger |
| 7 | Try Your Own Data | **Load examples**, then **Analyse** | A plan for three incidents |

---

## 3. The top of every page

### The railway-line menu

Each page is a stop on a line. The small train shows where you are. Click a stop to travel there. On a phone, swipe the line sideways.

| Stop | What it is for |
|---|---|
| Dashboard | Summary |
| 1 Agent & Search | Finding routes |
| 2 Data & ML | The incident records |
| 3 Neural Networks | Predicting risk |
| 4 Smart System | Safer routes |
| Try Your Own Data | Your own incidents |
| Logs | Messages from the system |

### Status

| You see | Meaning |
|---|---|
| Green **API online** | Ready |
| Orange **Task running** | Busy; wait before starting something else |
| Red **API offline** | The backend is not running |

### Theme

Choose **Dark**, **Light** or **System** (follows your computer). Your choice is remembered.

### Clock

The current time on your computer. It is hidden on phones.

### Common signs

| You see | Meaning |
|---|---|
| Green, tick | Low risk or success |
| Orange, triangle | Medium risk, warning or corrected value |
| Red, stop sign | High risk or error |
| Moving stripes | Loading |
| Train moving along a track | Progress of a task |
| Spinning circle on a button | Working |
| **–** or **n/a** | No value yet |
| Underlined words in a row | Tabs; click to switch view |

---

## 4. How to read the station map

The map has 15 rows and 15 columns, numbered from 0. Squares are written as **(row, column)**, counted from the top left, so (13, 1) is near the bottom left.

> Forms and CSV files use **Grid X** for the column and **Grid Y** for the row.

| You see | Meaning |
|---|---|
| Plain square | Easy to cross |
| Grey dot | Slower to cross; a bigger dot is slower |
| Diagonal stripes | Wall |
| Dashed square | A wall the vehicle has not noticed yet |
| Circle with **S** | Start |
| Star | Destination |
| Small train | The vehicle |
| Solid line | Route |
| Dotted line | Normal, planned or original route (see the key below the map) |
| Filled colour | A square a search method checked |
| Orange triangle | Medium risk |
| Red square | High risk |
| Red box with X | Discovered wall or new obstruction |
| Coloured ring | An incident |

Hold the mouse over a square to see its position and cost.

---

## 5. Playback controls

| Control | What it does |
|---|---|
| Curved arrow | Back to the start |
| Left arrow | One step back |
| Large button | Play or pause |
| Right arrow | One step forward |
| Double arrow | Jump to the end |
| 0.5x, 1x, 2x, 4x | Speed |
| Number on the right | Current step and total |

---

## 6. Dashboard

### Departure board

One row per module. Click a row to open it.

| Status | Meaning |
|---|---|
| Ready | Results are available (Module 1 is always Ready) |
| Not run | No saved results yet |
| Waiting, Running, Done | Shown during the full demo |

### Run full demo

Runs all four modules in order. **It trains the models again and replaces saved results.**

### Four numbers

| Number | Meaning | Example |
|---|---|--:|
| Incidents on record | Records in the dataset | 360 |
| Risk predicted correctly | Best network, on incidents it had not seen | 94.4% |
| Response-time error | Average miss of Module 2's time prediction | 2.23 min |
| Risk avoided | Risk reduction on the Normal layout, in the experiments | 100% |

"Risk avoided" does not mean danger is always avoided.

### Other sections

- **How serious are they?** Incidents split into Low, Medium and High.
- **What goes wrong:** number of incidents of each type.
- **Saved models:** *Saved* or *Missing* for each model file.

---

## 7. Agent & Search

Two tabs at the top: **Algorithm race** and **Agent playback**.

### 7.1 Algorithm race

The search starts automatically when you choose a layout.

| Option | What it does |
|---|---|
| Normal, Obstacle-Heavy, Unreachable-Goal | Choose a map |
| Edit grid | Change the map yourself |
| Show costs / Hide costs | Show the cost of each square |
| Run search | Run all three methods |

**While editing:**

| Option | What it does |
|---|---|
| Wall | Click a square to add or remove a wall |
| Start | Click a square to move the start |
| Goal | Click a square to move the destination |
| Reset layout | Undo your changes |
| Done editing | Return to the race, which runs again |

**Reading the race.** Colour spreads as each method checks squares, then its route appears. The table compares:

| Column | Meaning | Better |
|---|---|---|
| Path length | Squares in the route | Lower |
| Total cost | Difficulty of the route | **Lower** |
| Nodes expanded | Squares checked | Lower |
| Time | Calculation time | Lower |

The trophy marks the best method. It is chosen by total cost first. If the destination cannot be reached, you see **No path exists**.

### 7.2 Agent playback

Choose a scenario; it plays automatically.

| Scenario | What to watch |
|---|---|
| 1. Normal movement | A simple journey |
| 2. Neighbouring cell is blocked | It goes around a wall |
| 3. Reaching the incident location | It is already there |
| 4. No safe neighbouring move | It is surrounded and raises an alert |
| 5. Re-planning after a hidden hazard | It discovers walls and re-plans twice |
| 6. Target is unreachable | It explores, then raises an alert |

| Panel | Meaning |
|---|---|
| Position | Current square |
| Route cost | Cost so far |
| Re-plans | How many times it changed its plan |
| Walls known | Walls discovered |
| Percept | What it sensed in the four squares beside it |
| Selected action | What it decided; orange means it re-planned |

The final box shows whether it arrived and its score. Scenario 5 scores 80.

---

## 8. Data & ML

### Run Module 2 / Run again

Cleans the data and trains the KNN methods again. It replaces Module 2's saved results.

### Four numbers

| Number | Meaning | Example |
|---|---|--:|
| Clean records | Usable records | 360 |
| Values corrected | Impossible or extreme values marked as missing | 7 |
| KNN classifier F1 | Quality of risk prediction, treating all three levels equally; 1 is perfect | 0.588 |
| Response-time MAE | Average miss | 2.23 min |

### Data cleaning report

Shows which values were wrong, and in which record.

### Exploratory charts

| Tab | Shows |
|---|---|
| Risk | How many incidents of each level |
| Response time | How long responses take |
| Correlation | Which values rise and fall together; 1 is strongly together, -1 is opposite, 0 is unrelated |
| Boxplots | The usual range of each value; red dots are unusual |
| Elbow | Helps choose the number of location groups (3 are used) |

### Location clusters (K-Means)

Each dot is an incident. Numbered circles are the centres of the three areas. **Click anywhere** to see which area that point joins. An area number is not a danger level.

### KNN tuning

Compares values of `k` (how many similar incidents to look at) and two ways of measuring similarity, *euclidean* and *manhattan*. Solid lines are F1; dashed lines are accuracy. Higher is better.

### Raw dataset

| Option | What it does |
|---|---|
| Search box | Find records |
| Show only problem rows | Show records with corrected or missing values |
| Column name | Sort; click again to reverse |
| Prev / Next | Change page |

---

## 9. Neural Networks

### Train models / Train again

Trains both networks, usually in about a minute. **It replaces the saved models** and cannot be cancelled.

During training, **Live training** shows accuracy (should rise) and loss, meaning error (should fall). Solid lines are the lessons; dashed lines are the practice tests.

### Model comparison

Three methods on 72 unseen incidents. Green marks the best value in each column.

| Column | Meaning (1 is best) |
|---|---|
| Accuracy | Share of correct answers |
| Precision | When it names a level, how often it is right |
| Recall | How many incidents of each level it finds |
| Macro F1 | Balance of precision and recall across all levels |
| ROC-AUC | How well it separates the levels; 0.5 is guessing |

### Training curves

Tabs: **Accuracy** and **Loss**.

### ROC curves

Tabs: **Low**, **Medium**, **High**. A line nearer the top left is better. The dashed diagonal is guessing.

### Confusion matrices

Rows are the true level; columns are the prediction. Numbers on the diagonal, from top left to bottom right, are correct.

### Wrong predictions

Up to six mistakes. **Red** means risk was under-estimated, which is the dangerous kind. **Orange** means it was over-estimated.

### Predict a new incident

| Option | Values |
|---|---|
| Grid X, Grid Y | 0–14 |
| Incident type | Structural, Medical, Crowd Control, Electrical, Equipment Failure, Fire Smoke |
| Crowd density | 0–100% |
| Hazard severity | 1 (minor) to 5 (severe) |
| Equipment health | 0% (poor) to 100% (perfect) |
| Obstruction level | 0 (clear) to 3 (blocked) |
| Hour of day | 00:00–23:00 |

Press **Predict risk**. You get the predicted level, the chance of each level and a response time. With the starting values, the saved models give **High, 98.5%, 17.6 min**. A high percentage means the model is confident, not that it is certainly right.

---

## 10. Smart System

This page needs the models from the Neural Networks page. If they are missing, a button takes you there.

### Normal vs risk-aware routing

| Option | What it does |
|---|---|
| Predicted risk | Show the danger the network expects |
| True risk | Show the real danger |
| Drive risk-aware | Send the train along the safer route |
| Drive normal | Send it along the ordinary route |

The dotted line ignores danger; the solid line avoids *predicted* danger. The boxes compare length, travel cost and actual time. Actual time adds 2 minutes for each truly Medium square crossed and 6 for each High one.

### Automatic re-planning

The vehicle drives, something blocks the way, and it finds a new route **from where it is**. Move **Obstruction appears after step** to choose when.

### Multi-target planning

Four incidents, ranked. The first is handled first. **Re-rank** replays the sorting; it does not choose new incidents.

### Experiments

**Run experiments / Run again** repeats the 30 experiments and replaces their saved results.

| Option | What it shows |
|---|---|
| Performance | Overall score; higher is better |
| Risk | Danger met; lower is better |
| Response time | Minutes; lower is better |
| Averages | One row per layout and mode |
| All runs | Every run, after you run the experiments in this session |

The Unreachable-Goal layout scores 0 in both modes because no route exists. That is expected.

---

## 11. Try Your Own Data

### Step 1: Add incidents

**Form.** Fill in the fields and press **Add incident**. Name, crowd, equipment and obstruction may be left blank.

**CSV upload.** Press **Choose file**, or drop a `.csv` file on the box. **Download template** gives you a ready-made file to fill in. The limit is 200 incidents.

| Column | Required? | Values |
|---|:-:|---|
| `incident_id` | No | Up to 40 characters, no repeats |
| `incident_type` | Yes | Structural, Medical, Crowd_Control, Electrical, Equipment_Failure, Fire_Smoke |
| `grid_x`, `grid_y` | Yes | Whole number, 0–14 |
| `crowd_density_pct` | No | 0–100 |
| `hazard_severity_1_5` | Yes | Whole number, 1–5 |
| `equipment_health_pct` | No | 0–100 |
| `obstruction_level_0_3` | No | Whole number, 0–3 |
| `hour_of_day` | Yes | Whole number, 0–23 |

### Step 2: Set up the station

| Option | What it does |
|---|---|
| Normal, Obstacle-Heavy, Unreachable-Goal | Choose a map |
| Move start | Click a square to move the start |
| Add / remove walls | Click a square to change it |
| Reset map | Undo your changes |

### Step 3: Check and analyse

| Option | What it does |
|---|---|
| Load examples | Add three sample incidents (shown when the list is empty) |
| Any box in the table | Click to edit |
| Bin | Remove that row |
| Clear all | Remove every row |
| Analyse | Make predictions and plan routes |

Rows with mistakes are marked red and left out, with the reason below the table. Changing any input clears the old results, so press **Analyse** again.

### Your results

| You see | Meaning |
|---|---|
| Number in a box | Order to respond; **–** means unreachable |
| Coloured bar and percentages | Chance of Low, Medium and High |
| Response | Predicted minutes |
| Area | Location group |
| Risk-aware / Normal | Which route the train drives |
| Visit all | One route to every reachable incident, in order |
| Travel cost | Difficulty of the route |
| With risk delays | Travel cost plus predicted delays |
| Predicted risk | Danger expected on the route, from your other incidents |
| Download results | Save as a CSV file |

These are predictions. They cannot be marked right or wrong, because your incidents have no known answers.

---

## 12. Logs

Messages printed by the system, mostly useful if something goes wrong.

| Option | What it does |
|---|---|
| Filter lines | Show only lines containing your words |
| Pause / Resume | Stop or continue updating |
| Auto-scroll | Follow the newest line |
| Clear | Empty the screen; nothing else is deleted |

Red lines mention errors, orange lines warnings, and green lines completed or saved work. The colours are chosen by words in the line, so a red line is not always a problem.

---

## 13. Which buttons change saved results?

| Button | Replaces |
|---|---|
| Run full demo | All saved models and results |
| Run Module 2 / Run again | Module 2 models and results |
| Train models / Train again | Neural networks, Module 3 KNN and results |
| Run experiments / Run again (Smart System) | Experiment results |

Searching, agent playback, predicting, clicking the location map and analysing your own data do not change the models or the saved results shown on the Dashboard. Opening Smart System also refreshes two route pictures in `outputs/`.

---

## 14. Messages and what to do

| Message | What to do |
|---|---|
| API offline / Cannot reach the backend | Run `start_ui.bat` again |
| Another task is running | Wait until the status is **API online** |
| No Module 2 results yet | Press **Run Module 2** |
| Models not trained in this session | Press **Train models** |
| Module 3 models are missing | Train on the Neural Networks page |
| No path exists | The destination is walled in; change the map |
| The start / goal cell cannot be blocked | Move it, or reset the layout |
| Missing column(s) | Compare your file with the template |
| Please choose a .csv file | Save the file as CSV in Excel |
| None of the incidents are valid | Fix the boxes marked red |
| The incident is inside a wall | Move the incident or remove the wall |
| No route exists from the start | Remove a wall or move the start |

---

## 15. Questions

**What is kept when I change page?** Your own incidents and the latest module results. A map edited on Agent & Search is not kept.

**What is lost when I refresh?** Your own incidents and the full list of experiment runs. Saved Module 2 and 3 results and experiment averages come back.

**Is my data sent anywhere?** Your incidents are handled on this computer. The website does load its fonts from Google.

**Does it work on a phone?** The pages fit a small screen, but as set up, the app opens only on the computer running it.

**Can I turn off the animation?** Turn on "reduce motion" in your computer's settings. Most animation is then removed or shortened.

**Can I trust it in a real emergency?** No. It is a learning project, and its limits are listed in the [README](../README.md#limits).
