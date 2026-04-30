# 🔥 FIRE DATE PREDICTION SYSTEM - COMPLETE TRANSFORMATION

## 📋 Overview

Your wildfire prediction system has been **completely transformed** from a risk-based system to a **date-based prediction system**.

### ✨ What Changed

**BEFORE:**
- Predicted: "Will fire occur tomorrow?" (Yes/No)
- Output: Risk levels (LOW/MEDIUM/HIGH)
- Model: Binary classification

**AFTER:**
- Predicts: **"WHEN will fire occur?"** (Specific date)
- Output: **Fire dates** (e.g., "Fire on 2026-04-27")
- Prediction window: **1-7 days ahead**
- Model: LightGBM Regression (kept as requested)

---

## 🗂️ Files Updated

### **Backend (Python)**

1. **training.py** ✅ COMPLETELY REWRITTEN
   - Changed from binary classification to regression
   - Predicts `days_until_fire` (0-7 days)
   - New label generation logic
   - Saves model as `lgbm_fire_date_model.pkl`

2. **risk_map.py** ✅ COMPLETELY REWRITTEN
   - Generates fire date predictions instead of risk levels
   - Creates GeoJSON with predicted fire dates
   - Outputs urgency levels (CRITICAL/HIGH/MEDIUM/LOW)
   - Saves to `fire_dates_all.geojson`

3. **api.py** ✅ COMPLETELY REWRITTEN
   - New endpoints for fire date predictions
   - `/predictions/today` - Get today's predictions
   - `/predictions/timeline` - Get 7-day fire timeline
   - `/predict/location` - Predict for specific location
   - `/geojson` - Get map data

4. **fetch_firms.py** ✅ NO CHANGES
   - Kept as is (still fetches satellite data)

### **Frontend (HTML/CSS/JS)**

5. **index.html** ✅ COMPLETELY REDESIGNED
   - Modern, clean interface
   - 7-day fire timeline
   - Urgency summary cards
   - Display options panel

6. **app.js** ✅ COMPLETELY REWRITTEN
   - Displays fire dates on map
   - Color-coded urgency markers
   - Interactive popups with fire dates
   - Timeline visualization

7. **style.css** ✅ COMPLETELY REDESIGNED
   - Modern dark theme
   - Gradient backgrounds
   - Smooth animations
   - Responsive design

---

## 🚀 How to Use

### **Step 1: Train the New Model**

```bash
python training.py
```

**What it does:**
- Loads satellite fire data
- Creates new label: `days_until_fire` (0-7)
- Trains LightGBM regression model
- Saves model to `outputs/models/lgbm_fire_date_model.pkl`

**Expected output:**
```
✅ Training samples with fire within 7 days: XXXXX
Distribution of days until fire:
0    XXX
1    XXX
2    XXX
...
📊 MODEL PERFORMANCE
MAE (days): 1.XX
RMSE (days): 2.XX
Accuracy within ±1 day: XX.X%
```

### **Step 2: Generate Fire Date Map**

```bash
python risk_map.py
```

**What it does:**
- Loads trained model
- Predicts fire dates for all locations
- Creates GeoJSON with fire date predictions
- Saves to `outputs/riskmap/fire_dates_all.geojson`

**Expected output:**
```
✅ FIRE DATE MAP UPDATED
Observed date : 2026-XX-XX
Base date     : 2026-XX-XX
Prediction    : Fire dates for next 7 days
📊 URGENCY SUMMARY:
  CRITICAL: XX locations
  HIGH: XX locations
  MEDIUM: XX locations
  LOW: XX locations
```

### **Step 3: Run API Server**

```bash
uvicorn api:app --reload
```

**API Endpoints:**
- `http://localhost:8000/` - API info
- `http://localhost:8000/predictions/today` - Today's predictions
- `http://localhost:8000/predictions/timeline` - 7-day timeline
- `http://localhost:8000/geojson` - Map data

### **Step 4: View Dashboard**

Open `index.html` in your browser to see:
- 📅 7-day fire timeline
- ⚡ Urgency summary (Critical/High/Medium/Low)
- 🗺️ Interactive map with fire date predictions
- 🎨 Modern, dark-themed interface

---

## 📊 Understanding the Output

### **Urgency Levels**

| Level | Days Until Fire | Color | Meaning |
|-------|----------------|-------|---------|
| CRITICAL | 0 days | Red | Fire expected TODAY |
| HIGH | 1-2 days | Orange | Fire within 1-2 days |
| MEDIUM | 3-4 days | Yellow | Fire within 3-4 days |
| LOW | 5-7 days | Green | Fire within 5-7 days |

### **Map Markers**

- **Red circles** 🔴 = Observed fires (actual satellite detections)
- **Colored circles** 🟠🟡🟢 = Predicted fire locations (color = urgency)
- Click any marker to see:
  - Predicted fire date
  - Days until fire
  - Urgency level
  - Prediction confidence

### **Timeline Panel**

Shows daily fire counts for the next 7 days:
```
Today      2026-04-22    15 fires
Tomorrow   2026-04-23    8 fires
+2 days    2026-04-24    12 fires
...
```

---

## 🔧 Technical Details

### **Model Architecture**

```python
LGBMRegressor(
    objective="regression",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50
)
```

**Input Features:**
1. `fire_3d` - Fire count (last 3 days)
2. `frp_3d` - Fire Radiative Power (last 3 days)
3. `frp_max` - Maximum FRP
4. `fire_days_7d` - Days with fire (last 7 days)
5. `fire_yesterday` - Fire count yesterday
6. `frp_trend` - FRP trend
7. `bright_mean` - Average brightness
8. `confidence_mean` - Average confidence

**Output:**
- `days_until_fire` (0-7 days, or -1 for "no fire expected")

### **Label Generation Logic**

For each location and date, the model looks ahead 1-7 days:
- If fire detected → label = days until that fire
- If no fire in 7 days → label = -1 (excluded from training)

### **Prediction Confidence**

```python
confidence = 1 - |predicted_days - rounded_days|
```

Higher confidence when prediction is close to an integer (e.g., 2.05 days has higher confidence than 2.45 days)

---

## 📁 Output Files Structure

```
outputs/
├── models/
│   └── lgbm_fire_date_model.pkl    # Trained model
├── features/
│   └── full_features.csv            # Feature dataset
├── metadata/
│   └── dataset_info.json            # Model metadata
└── riskmap/
    ├── fire_dates_all.geojson       # Map data
    └── latest.json                  # Latest prediction info
```

---

## 🎨 UI Features

### **Sidebar Controls**

- **Base Date** - Shows current prediction base date
- **7-Day Timeline** - Daily fire count predictions
- **Urgency Summary** - Count by urgency level
- **Display Options:**
  - Toggle observed fires
  - Toggle predictions
  - Toggle marker clustering

### **Map Features**

- **Dark theme** for better visibility
- **Marker clustering** for performance
- **Interactive popups** with fire details
- **Color-coded urgency** markers
- **Legend** showing urgency levels

---

## 🔄 Workflow Summary

```
1. fetch_firms.py   → Downloads satellite data
2. training.py      → Trains fire date prediction model
3. risk_map.py      → Generates fire date predictions
4. api.py           → Serves predictions via API
5. index.html       → Displays interactive dashboard
```

---

## 📈 Performance Metrics

The model is evaluated using:
- **MAE (Mean Absolute Error)** - Average prediction error in days
- **RMSE (Root Mean Squared Error)** - Penalizes larger errors
- **Accuracy within ±1 day** - % of predictions within 1 day of actual

Target performance:
- MAE < 2 days
- Accuracy within ±1 day > 70%

---

## 🐛 Troubleshooting

### "Model not found" error
- Run `training.py` first to create the model

### "GeoJSON not found" error
- Run `risk_map.py` to generate predictions

### Empty map
- Check if `fire_dates_all.geojson` exists
- Verify file path in `app.js`

### No predictions showing
- Ensure model has been trained
- Check browser console for errors
- Verify API server is running

---

## 🎯 Next Steps

1. **Train the model** with your data
2. **Generate predictions** for fire dates
3. **Run the API** server
4. **Open the dashboard** to visualize predictions
5. **Monitor accuracy** and retrain as needed

---

## ✅ Summary

Your system now predicts **WHEN fires will occur** (specific dates within 7 days) instead of just risk levels. The new interface clearly shows:

- **Fire dates** (e.g., "Fire on 2026-04-27")
- **Urgency levels** (Critical/High/Medium/Low)
- **7-day timeline** with daily fire counts
- **Interactive map** with color-coded predictions

**All files are ready to use!** Just follow the steps above to start predicting fire dates. 🔥
