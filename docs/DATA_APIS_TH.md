# แหล่งข้อมูลจริงและ API ที่โปรเจกต์ใช้

เอกสารนี้สรุปว่า **ดึงข้อมูลจากไหน เรียกอย่างไร ต้องมี key หรือไม่** — ใช้ประกอบรายงานหรือต่อยอดระบบได้

> พิมพ์เป็น PDF: เปิดไฟล์ในเบราว์เซอร์ → พิมพ์ → บันทึกเป็น PDF

---

## 1. NASA FIRMS (จุดร้อน VIIRS NRT)

| รายการ | รายละเอียด |
|--------|------------|
| **หน้าที่** | จุดร้อนจากดาวเทียม VIIRS แบบใกล้เวลาจริง (NRT) — แกนกลางของโมเดลและแผนที่ |
| **การยืนยันตัวตน** | ต้องมี **`FIRMS_API_KEY`** (Map Key) ใน `.env` |
| **สคริปต์ในโปรเจกต์** | `src/fetch_firms.py` |
| **ผลลัพธ์** | `data/firms/firms_all.parquet` (สะสม) |
| **เรียก API อย่างไร** | สคริปต์ใช้ `requests` ไปที่เซิร์ฟเวอร์ NASA ตามเอกสาร FIRMS (ดูใน `fetch_firms.py`) — ไม่ต้องเรียกมือถ้ารันสคริปต์ |
| **ตรวจสถานะ key** | เปิด `https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=...` |

---

## 2. Open-Meteo Archive (สภาพอากาศ ERA5)

| รายการ | รายละเอียด |
|--------|------------|
| **หน้าที่** | อุณหภูมิ / ฝน / ลม / ET₀ รายวัน (reanalysis จริง) ต่อพิกัด |
| **การยืนยันตัวตน** | **ไม่ต้องมี API key** |
| **ฐาน URL หลัก** | สคริปต์ใช้ endpoint แบบ **Historical / Archive** (ดู `ARCHIVE_URL` ใน `fetch_weather.py`) |
| **สคริปต์** | `src/fetch_weather.py` |
| **ผลลัพธ์** | `data/weather/weather_cache.parquet` |
| **ข้อจำกัด** | มี **rate limit** (HTTP 429) — ลด worker (`OPEN_METEO_MAX_WORKERS`), ใช้ `--sleep`, หรือ `--quiet-hours` ตาม docstring |
| **เรียก API อย่างไร** | GET พารามิเตอร์ `latitude`, `longitude`, `start_date`, `end_date`, `daily=...`, `timezone=Asia/Bangkok` — รายละเอียดอยู่ในฟังก์ชันดึง archive ใน `fetch_weather.py` |

---

## 3. GISTDA (สถานีอวกาศไทย — ArcGIS REST)

| รายการ | รายละเอียด |
|--------|------------|
| **หน้าที่** | จุดร้อน VIIRS NPP + MODIS ที่ GISTDAประมวลผลซ้ำ — มี **`lu_name`** (ประเภทการใช้ประโยชน์ที่ดินไทย) และชื่อจังหวัด/อำเภอ/ตำบล |
| **การยืนยันตัวตน** | **ไม่ต้องมี API key** (บริการสาธารณะ) |
| **ฐาน URL** | `https://gistdaportal.gistda.or.th/data/rest/services/FR_Fire` |
| **เลเยอร์ที่ใช้** | `hotspot_npp_daily/MapServer/0` (VIIRS NPP), `hotspot_daily/MapServer/0` (MODIS) |
| **สคริปต์** | `src/fetch_gistda_hotspots.py` |
| **ผลลัพธ์** | `data/gistda/gistda_hotspots.parquet` |
| **เรียก API อย่างไร** | ArcGIS **MapServer** — สคริปต์ใช้ `query` แบบ `where`, `geometry`, `outFields`, `f=json` (ดูใน `fetch_gistda_hotspots.py`) |

### GISTDA LULC (ป่า / เกษตร เป็นพื้นที่)

| รายการ | รายละเอียด |
|--------|------------|
| **สคริปต์** | `src/fetch_gistda_lulc.py` |
| **ผลลัพธ์** | `data/static/gistda_lulc_per_cell.parquet` |
| **ฐาน URL** | `https://gistdaportal.gistda.or.th/data/rest/services/Mhesi/...` |

### สถานะใน pipeline เทรน

- **ขณะนี้** ข้อมูล GISTDA ที่ดึงได้ถูกเก็บเป็น **แคชแยก** — **ยังไม่ได้ join เข้า `train.py` / ฟีเจอร์อัตโนมัติ** (ต้องมีขั้นตอนใน `data_loader.py` + `features.py` หากจะให้โมเดลใช้ `lu_name` หรือ LULC จริง)
- การรัน `./run.sh --gistda` จะ **อัปเดตแคช** ให้พร้อมใช้วิเคราะห์ / ต่อยอดโค้ดภายหลัง

---

## 4. แหล่งอื่นที่เกี่ยวข้อง (ไม่ใช่ REST แบบเดียวกับบน)

| แหล่ง | สคริปต์ | หมายเหตุ |
|--------|---------|----------|
| **Hansen Global Forest Change** | `fetch_treecover.py` | ดึงแรสเตอร์ผ่าน HTTP range read → `tree_cover_per_cell.parquet` |
| **ข้อมูล raw ประวัติ FIRMS** | วางใน `data/raw/` | โหลดผ่าน `data_loader` ร่วมกับ `firms_all` |

---

## 5. คำสั่งสั้น ๆ ที่ใช้บ่อย

```bash
cd src
python fetch_firms.py --days 5              # NASA FIRMS → firms_all
python fetch_weather.py                     # Open-Meteo (ช้าได้ถ้าโดน 429)
python fetch_gistda_hotspots.py             # GISTDA จุดร้อน → gistda_hotspots.parquet
python fetch_gistda_lulc.py                 # GISTDA พื้นที่ป่า/เกษตร (รันครั้งเดียวยาว)
```

จากรากโปรเจกต์ (มี `.venv`):

```bash
./run.sh --fresh --weather --gistda --fast   # ดึง FIRMS + weather + GISTDA แล้วเทรนแบบเร็ว (LGBM เท่านั้น)
```

---

## 6. จำกัด “ความยาวประวัติวัน” ตอนเทรน (`max_history_days`)

| รายการ | รายละเอียด |
|--------|------------|
| **CLI** | `python train.py --max-history-days 180` — เก็บเฉพาะ **N วันปฏิทินล่าสุด** นับจากวันที่ใหม่สุดในข้อมูล densified (ก่อนสร้างฟีเจอร์) |
| **ENV** | `MAX_TRAIN_HISTORY_DAYS` — ใช้เมื่อไม่ส่ง CLI (ค่า `0` = ไม่ตัด) |
| **ผลต่อความเร็ว** | แถวน้อยลง → สร้างฟีเจอร์และ tune เร็วขึ้นชัดเจน |
| **ผลต่อ MAE / accuracy** | **ไม่การันตีดีขึ้น** — ตัดข้อมูลเก่าอาจช่วยถ้า “ระบบจริง” เปลี่ยนไปจากอดีตไกล ๆ แต่ถ้าตัดมากเกินไปจะเสีย **ฤดูกาล / แพทเทิร์นระยะยาว** และ test split อาจสั้นลงจน metric ผันผวน |

ค่า `max_history_days_applied` ถูกบันทึกใน `outputs/metadata/dataset_info.json` หลังเทรนเต็ม

---

*อัปเดตตามโค้ดใน repo — ถ้า URL หรือพารามิเตอร์ฝั่งผู้ให้บริการเปลี่ยน ให้ดูที่ไฟล์ `fetch_*.py` เป็นหลัก*
