# RailSentry: Code Guide

This guide explains every code file to someone seeing the project for the first time. **You do not need to know programming.** You never need to edit a file to use RailSentry.

For how to use the website, see the [App guide](APP_GUIDE.md). For installation, see the [README](../README.md).

## Contents

1. [The idea in one minute](#1-the-idea-in-one-minute)
2. [Words you will meet](#2-words-you-will-meet)
3. [How the files work together](#3-how-the-files-work-together)
4. [The Python files](#4-the-python-files)
5. [The website files](#5-the-website-files)
6. [Data, models and results](#6-data-models-and-results)
7. [Follow one click from start to finish](#7-follow-one-click-from-start-to-finish)
8. [Which file should I open?](#8-which-file-should-i-open)

---

## 1. The idea in one minute

Imagine a small emergency vehicle inside a railway station. The station is drawn as a board of 15 × 15 squares, like a large chessboard. Some squares are walls, some are slow to cross, and some are dangerous because something has gone wrong there.

RailSentry has four main jobs, and each has its own file, called a **module**:

| Module | Question it answers | Everyday comparison |
|---|---|---|
| 1 | How do I get there? | A sat-nav |
| 2 | What can past incidents teach us? | A records office |
| 3 | How dangerous is a new incident? | An experienced inspector |
| 4 | How do I get there *safely*? | A sat-nav that listens to the inspector |

---

## 2. Words you will meet

| Word | Meaning |
|---|---|
| **Code** | Written instructions that a computer follows |
| **File** | One document of instructions |
| **Function** | A small named job inside a file, such as "find the cheapest route" |
| **Python** | The language used for the AI part |
| **TypeScript / React** | The language and toolkit used for the website |
| **Frontend** | What you see in the browser |
| **Backend** | The part that does the work out of sight |
| **API** | The messenger between the two |
| **Dataset** | A table of past records |
| **CSV** | A simple table file that Excel can open |
| **Model** | Something that has learned from examples and can now make predictions |
| **Training** | Showing examples to a model so it learns |
| **Prediction** | A model's educated guess |
| **Grid** | The 15 × 15 station map |
| **Cost** | How hard a square is to cross; a higher number is slower |

---

## 3. How the files work together

```text
   You
    |
    v
Website (frontend folder)          Terminal menu (main.py)
    |                                        |
    v                                        |
api.py (the messenger)                       |
    |                                        |
    +--------------------+-------------------+
                         v
   Module 1     Module 2     Module 3     Module 4
   routes       data         predictions  everything together
                         |
                         v
        dataset, saved models, saved results
```

There are two ways in: the website, or a plain text menu. **Both use the same four modules**, so they follow the same logic.

Some modules need others:

- Module 3 uses Module 2's cleaned data.
- Module 4 uses the maps from Module 1, the data from Module 2 and the predictions from Module 3.
- `user_data.py` uses Modules 2, 3 and 4.

---

## 4. The Python files

### 4.1 `main.py`: the front door

**Purpose:** shows a text menu and starts whichever module you choose.

| Choice | What happens |
|:-:|---|
| 1–4 | Opens that module |
| 5 | Runs all four modules in order |
| 6 | Closes the program |

If one module fails, the menu shows the error and stays open.

### 4.2 `module1_agent_search.py`: finding the way

**Purpose:** builds the station maps, finds routes and controls the vehicle.

**The maps.** Three are built in: *Normal*, *Obstacle-Heavy* (many walls) and *Unreachable-Goal* (the destination is walled in on purpose).

**The three search methods:**

| Method | How it searches | Result |
|---|---|---|
| **BFS** | Spreads out evenly, like ripples in water | Fewest squares |
| **DFS** | Follows one path as far as it can, then backs up | Any route, often not a good one |
| **A\*** ("A star") | Considers both the cost so far and the distance left | Cheapest route on these maps |

**The agent** is the vehicle's "brain". It knows how slow each square is, but not where the walls are. It senses only the four squares beside it, remembers any walls it finds, and then:

1. If it has arrived, it raises an arrival alert.
2. If it is completely surrounded, it raises an alert.
3. If the target is next to it, it moves there.
4. If its route is blocked, it makes a new plan.
5. If no route is left, it reports that the target is unreachable.
6. Otherwise, it takes the next step.

**Its score:** 100 points for arriving, minus the route cost, minus 5 for each new plan, minus any waiting.

| Main function | Job |
|---|---|
| `create_layouts` | Builds the three maps |
| `bfs_search`, `dfs_search`, `astar_search` | The three methods |
| `RailwayResponseAgent` | The vehicle's brain |
| `build_agent_scenarios` | The six situations shown on the website |
| `visualise_grid` | Saves a picture of a map |

### 4.3 `module2_data_ml.py`: learning from records

**Purpose:** cleans the 360 records, draws charts and trains simple prediction methods.

**Cleaning.** Some records contain mistakes, such as 190% crowd density. The file marks impossible values as missing. It found 4 impossible crowd values and 3 extreme response times (95, 120 and 150 minutes). Missing inputs are later filled with a typical value.

**Location groups (K-Means).** Splits incident locations into three groups by position. A group is only an area of the station; it does not mean safe or dangerous.

**KNN ("nearest neighbours").** To judge a new incident, it looks at the most similar past ones. If the most similar cases were mostly High risk, it predicts High. `k` is how many similar cases it looks at.

| Main function | Job |
|---|---|
| `load_prepared_data` | Gives the other modules the cleaned data |
| `handle_missing_values`, `handle_outliers` | Cleaning |
| `create_location_clusters` | Makes the three groups |
| `train_knn_classifier` | Predicts risk |
| `train_knn_regressor` | Predicts response time |
| `draw_required_graphs` | Saves charts |

### 4.4 `module3_neural_network.py`: predicting risk

**Purpose:** trains two neural networks to predict Low, Medium or High risk.

A **neural network** is made of layers of small calculators. It guesses, checks its mistake, adjusts, and repeats. Each full pass through the examples is an **epoch**.

The records are divided like a school course:

| Part | Records | Like |
|---|--:|---|
| Training | 230 | Lessons |
| Validation | 58 | Practice tests, used to choose the better network |
| Test | 72 | The final exam, seen only at the end |

Training stops early when practice results stop improving. Mistakes on rare High-risk incidents count more, because there are few examples.

| Network | Design |
|---|---|
| Architecture 1 | Layers of 64 and 32 units |
| Architecture 2 | Layers of 128, 64 and 32, with some units randomly switched off while learning so it does not simply memorise |

In the saved results, Architecture 2 was chosen. It got 94.4% of the final exam right.

| Main function | Job |
|---|---|
| `prepare_training_data` | Divides the records |
| `train_ann_models` | Trains both networks |
| `train_and_evaluate_knn` | Trains KNN for comparison |
| `predict_incident` | Predicts one new incident |
| `analyse_wrong_predictions` | Lists its mistakes |

### 4.5 `module4_integrated_system.py`: the safer route

**Purpose:** combines everything.

1. Takes real incidents and asks the network how risky each one is.
2. Marks the predicted danger on the map.
3. Finds a normal route (ignores danger) and a risk-aware route (avoids *predicted* danger).
4. Scores both routes using the *true* risk, so the comparison is fair.
5. Makes something block the route, and re-plans from the vehicle's position.
6. Repeats this 10 times on each of the three maps.

| | Medium risk square | High risk square |
|---|--:|--:|
| Extra cost when planning | +4 | +10 |
| Delay when actually crossed | +2 minutes | +6 minutes |

These four numbers were chosen for the project.

**Priority score:** `20 × risk − 0.5 × response time − 0.5 × route cost`, where Low = 0, Medium = 1 and High = 2. The highest score is handled first.

| Main function | Job |
|---|---|
| `load_incidents_with_predictions` | Adds predictions to the records |
| `compare_routing` | Normal against risk-aware |
| `demonstrate_replanning` | Blocks the route and re-plans |
| `demonstrate_multi_target` | Ranks four incidents |
| `plan_user_incidents` | Routes for incidents you enter |
| `run_experiments` | Runs the 30 experiments |

### 4.6 `user_data.py`: checking your data

**Purpose:** handles the *Try Your Own Data* page.

It reads your form entries or CSV, checks **every** value, and explains each problem clearly. Good rows are sent to the models; rows with mistakes are left out. Nothing you enter is used for training.

### 4.7 `api.py`: the messenger

**Purpose:** lets the website talk to the Python modules.

Think of a waiter: the website places an order, `api.py` takes it to the kitchen (the modules) and brings back the result. It also:

- runs long jobs in the background and reports their progress
- allows only one heavy job at a time
- keeps printed messages for the Logs page
- saves the latest results, so the Dashboard can show them later

### 4.8 `tests/test_user_data.py`: the safety check

**Purpose:** 18 automatic checks on things like empty files, impossible values, walls and unreachable incidents. If a future change breaks something they cover, they fail.

### 4.9 Small helper files

| File | Purpose |
|---|---|
| `requirements.txt` | Shopping list of Python packages |
| `start_ui.bat` | Double-click to start the website |

---

## 5. The website files

Everything is inside `frontend/`.

### 5.1 Pages: `src/pages/`

One file for each stop in the menu.

| File | Page |
|---|---|
| `Dashboard.tsx` | Dashboard |
| `AgentSearch.tsx` | Agent & Search |
| `DataML.tsx` | Data & ML |
| `NeuralNetworks.tsx` | Neural Networks |
| `SmartSystem.tsx` | Smart System |
| `TryData.tsx` | Try Your Own Data |
| `Logs.tsx` | Logs |

### 5.2 Building blocks: `src/components/`

Parts reused on many pages.

| File | Purpose |
|---|---|
| `Layout.tsx` | Top of every page: name, railway-line menu, status, theme, clock |
| `ui.tsx` | Buttons, tabs, titles, labels, progress bar, messages |
| `StationGrid.tsx` | The station map and moving train |
| `PlaybackControls.tsx` | Play, pause, step and speed buttons |
| `charts.tsx` | Shared chart style and the coloured number tables |
| `reactbits/SplitFlapText.tsx` | The flipping departure-board letters |
| `reactbits/CountUp.tsx` | The rolling numbers |
| `reactbits/LICENSE.md` | Permission terms for those two parts; keep it |

### 5.3 Behind the scenes: `src/lib/`

| File | Purpose |
|---|---|
| `api.ts` | Sends requests to `api.py` |
| `useJob.ts` | Follows a long job's progress |
| `usePlayback.ts` | Counts the steps of an animation |
| `store.tsx` | Remembers results when you change page |
| `theme.tsx` | Dark and light colours |

### 5.4 Setup files

You will almost never need these.

| File | Purpose |
|---|---|
| `index.html` | The empty page the website loads into |
| `src/main.tsx` | Starts the website |
| `src/App.tsx` | Connects each address to its page |
| `src/index.css` | Colours and basic style |
| `tailwind.config.js` | Style settings |
| `postcss.config.js` | Prepares the styles for browsers |
| `vite.config.ts` | Runs the website and passes requests to the backend |
| `tsconfig.json` | Rules for checking the code |
| `package.json` | Shopping list of website packages |
| `package-lock.json` | The exact version of each package; created automatically |
| `.gitignore` | Files that should not be shared |
| `node_modules/` | The downloaded packages; created by `npm install` |

---

## 6. Data, models and results

**`railway_station_incident_response_dataset.csv`** holds the 360 past incidents. Keep it: everything depends on it.

**`models/`** holds what the computer has learned. Training replaces these files.

| File | Contains |
|---|---|
| `ann_architecture_1.keras`, `ann_architecture_2.keras` | Both neural networks |
| `best_risk_model.keras` | The chosen network |
| `module3_preprocessor.joblib` | Prepares values for the network |
| `module3_risk_classes.joblib` | The Low / Medium / High order |
| `module3_knn_classifier.joblib` | Module 3 KNN for risk |
| `module3_knn_regressor.joblib` | Module 3 KNN for response time; used for new predictions |
| `knn_risk_classifier.joblib`, `knn_response_time_regressor.joblib` | Module 2 KNN |
| `location_kmeans.joblib`, `location_scaler.joblib` | Location groups |

**`outputs/`** holds pictures (`.png`), experiment tables (`.csv`) and `ui_state.json`, the results the website remembers.

**`__pycache__/`** is created by Python and can be ignored.

---

## 7. Follow one click from start to finish

You press **Predict risk** on the Neural Networks page:

1. `NeuralNetworks.tsx` collects the values you chose.
2. `api.ts` sends them to the backend.
3. `api.py` receives them and calls Module 3.
4. `module3_neural_network.py` checks them, loads the saved network and predicts the risk.
5. The saved KNN model predicts the response time.
6. `api.py` sends the answer back.
7. `NeuralNetworks.tsx` shows the result and fills the bars.

Every other button follows a similar journey.

---

## 8. Which file should I open?

| I want to understand... | Open |
|---|---|
| How routes are found | `module1_agent_search.py` |
| How mistakes in the data are handled | `module2_data_ml.py` |
| How risk is predicted | `module3_neural_network.py` |
| How danger is avoided | `module4_integrated_system.py` |
| How my own data is checked | `user_data.py` |
| How the website gets its answers | `api.py` |
| What a page looks like | The matching file in `frontend/src/pages/` |
