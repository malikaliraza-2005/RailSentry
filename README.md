# RailSentry

**AI-based smart railway station incident response and routing system**

RailSentry predicts how dangerous an incident in a railway station is, then finds a safer route for the response vehicle. It includes a website, a terminal version and four AI modules written in Python.

## The problem

When something goes wrong in a station, such as a fire or a medical emergency, the response team has to answer three questions quickly:

1. How serious is it?
2. Which incident should we handle first?
3. What is the safest way to get there?

RailSentry answers all three using 360 past incident records and a 15 × 15 map of a station.

## What it does

- **Finds routes** with three search methods (BFS, DFS and A\*) and compares them.
- **Simulates a vehicle** that sees only the squares beside it and changes its plan when it meets a wall.
- **Cleans and explores** the incident data.
- **Predicts risk** (Low, Medium or High) with a neural network.
- **Predicts response time** in minutes.
- **Plans safer routes** around predicted danger, and re-plans when the way is blocked.
- **Accepts your own incidents** through a form or a CSV file.

## Results

| Measure | Result |
|---|---|
| Risk predicted correctly (best neural network, 72 unseen incidents) | 94.4% |
| Same test, KNN method | 76.4% |
| Average response-time error (Module 3 KNN, used for new predictions) | 2.09 minutes |

Average over 10 experiments on each layout:

| Layout | Risk met without predictions | Risk met with predictions | Score without | Score with |
|---|--:|--:|--:|--:|
| Normal | 5.1 | 0.0 | 3.7 | 66.7 |
| Obstacle-heavy | 3.1 | 0.2 | 27.8 | 63.2 |
| Unreachable goal | No route | No route | 0 | 0 |

Your numbers may change slightly if you train the models again. See [Limits](#limits) before relying on these results.

## What you need

| Requirement | Notes |
|---|---|
| Windows 10 or 11 | `start_ui.bat` is a Windows file |
| [Python 3.12](https://www.python.org/downloads/) | The version used for this project (64-bit) |
| [Node.js](https://nodejs.org/) | Built with version 24; version 18 or newer is needed |
| About 3 GB of free space | TensorFlow is large |
| Internet | For installation, and for the website's fonts |

## Install

Open PowerShell in this folder (the one containing `main.py`). You only do this once.

**1. Create the Python environment.** Use this short path, because TensorFlow can fail to install in long folder paths.

```powershell
py -3.12 -m venv C:\venvs\ailab
C:\venvs\ailab\Scripts\python.exe -m pip install -r requirements.txt
```

**2. Install the website's packages.**

```powershell
cd frontend
npm install
cd ..
```

## Run the website

Double-click **`start_ui.bat`**. It opens two windows, then your browser at <http://127.0.0.1:5173>. Keep both windows open while you use the website, and close them to stop.

To start it by hand, use two PowerShell windows:

```powershell
# Window 1: backend
C:\venvs\ailab\Scripts\python.exe -m uvicorn api:app --port 8000
```

```powershell
# Window 2: frontend
cd frontend
npm run dev
```

## Run in the terminal

```powershell
C:\venvs\ailab\Scripts\python.exe main.py
```

Choose a module from the menu, or option 5 to run all four. Chart windows open one at a time; close each to continue.

Trained models are included, so you can use everything straight away. Training again replaces them.

## Run the tests

```powershell
C:\venvs\ailab\Scripts\python.exe -m unittest discover tests
```

## How it is built

```text
Browser (React website)  <-->  api.py (FastAPI)  <-->  Python modules  <-->  data and saved models
```

| Part | Technology |
|---|---|
| Search and agent | Python, NumPy |
| Data and machine learning | pandas, scikit-learn |
| Neural networks | TensorFlow / Keras |
| Backend | FastAPI, Uvicorn |
| Website | React, TypeScript, Vite, Tailwind CSS |
| Charts and animation | Recharts, Framer Motion, React Bits |

## Folder map

| Item | What it is |
|---|---|
| `main.py` | Terminal menu |
| `module1_agent_search.py` | Route finding and the agent |
| `module2_data_ml.py` | Data cleaning, charts, KNN, location groups |
| `module3_neural_network.py` | Neural networks and predictions |
| `module4_integrated_system.py` | Risk-aware routing and experiments |
| `user_data.py` | Checks and analyses incidents you provide |
| `api.py` | Connects the website to Python |
| `railway_station_incident_response_dataset.csv` | The 360 incident records |
| `requirements.txt` | Python packages to install |
| `start_ui.bat` | Starts the website |
| `frontend/` | The website |
| `models/` | Saved, trained models |
| `outputs/` | Charts and results |
| `tests/` | Automatic checks |
| `doc/` | The guides below |

## Guides

- [Code guide](doc/CODE_GUIDE.md): every file explained for someone with no technical background
- [App guide](doc/APP_GUIDE.md): every page, button and option on the website

## The data

Each incident record contains:

| Column | Meaning | Values |
|---|---|---|
| `incident_id` | Name of the record | e.g. `STN-0001` |
| `grid_x`, `grid_y` | Column and row on the map | 0–14 |
| `incident_type` | What happened | Structural, Medical, Crowd_Control, Electrical, Equipment_Failure, Fire_Smoke |
| `crowd_density_pct` | How crowded | 0–100 |
| `hazard_severity_1_5` | How severe | 1–5 |
| `equipment_health_pct` | Equipment condition | 0–100 |
| `obstruction_level_0_3` | How blocked | 0–3 |
| `hour_of_day` | Hour | 0–23 |
| `risk_level` | The answer to predict | Low, Medium, High |
| `response_time_min` | The answer to predict | Minutes |

## Limits

- This is a learning project, **not** a real safety system.
- The station is a simplified map, not a real building.
- 360 records is a small dataset, and only 36 are High risk. The unseen test had 72 incidents, so one mistake changes accuracy by about 1.4 points.
- The routing experiments include incidents the network learned from, so they are likely to look better than they would on entirely new incidents.
- The extra delays and penalties used to score routes were chosen for the project. They were not measured in a real station.
- Incidents you enter have no known answers, so their accuracy cannot be measured.

## Problems and fixes

| Problem | Fix |
|---|---|
| "Application Control policy has blocked this file" | Windows Smart App Control is blocking TensorFlow or scikit-learn. Use another computer or Google Colab for the modules, or review the setting in Windows Security. Turning it off lowers protection and may not be reversible. |
| An error about long paths during installation | Create the environment at `C:\venvs\ailab`, as shown above |
| The website says **API offline** | The backend window is closed or still starting. Run `start_ui.bat` again |
| Port 5173 or 8000 is already in use | Close the older RailSentry windows, then start again |
| `npm` is not recognised | Install Node.js, then open a new PowerShell window |
| "Saved models are missing" | Run Module 2, then train on the Neural Networks page |

## Credits

Split-flap lettering and rolling numbers come from [React Bits](https://reactbits.dev), under MIT + Commons Clause. Its licence is in `frontend/src/components/reactbits/LICENSE.md` and must stay with those files.
