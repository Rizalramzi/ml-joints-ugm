# EduAdapt AI — Machine Learning & Personalized Learning

EduAdapt AI adalah sistem pembelajaran adaptif berbasis **Machine Learning** dan **Generative AI** yang dirancang untuk membantu mengidentifikasi karakteristik belajar siswa, mengukur tingkat penguasaan materi, memberikan rekomendasi pembelajaran, serta menghasilkan *personalized lesson plan*.

Proyek ini dikembangkan sebagai bagian dari eksplorasi teknologi pendidikan dengan fokus pada:

- Student profiling
- Unsupervised learning
- Learning gap analysis
- Rule-based personalized recommendation
- Generative AI untuk pembuatan lesson plan
- REST API menggunakan FastAPI

> **Status proyek:** Tahap machine learning dan integrasi API sudah berjalan. Penyimpanan data menggunakan database masih direncanakan dan belum menjadi bagian dari implementasi saat ini.

---

## 1. Project Overview

Sistem EduAdapt AI menerima data performa belajar siswa, kemudian memprosesnya melalui beberapa tahap:

1. Membaca dan menyiapkan data siswa.
2. Melakukan preprocessing terhadap fitur numerik.
3. Mengelompokkan siswa menggunakan algoritma K-Means.
4. Membentuk profil belajar siswa.
5. Menghitung mastery score dan learning gap.
6. Menentukan prioritas pembelajaran.
7. Menghasilkan rekomendasi pembelajaran berdasarkan area penilaian terlemah.
8. Mengirim profil dan rekomendasi ke Gemini.
9. Menghasilkan lesson plan personal dalam format JSON.
10. Menyediakan hasil analisis melalui REST API.

### Tujuan Utama

EduAdapt AI tidak hanya memberikan rekomendasi yang sama untuk semua siswa. Sistem berusaha menyesuaikan rekomendasi berdasarkan:

- Nilai diagnostik
- Rata-rata tugas
- Rata-rata kuis
- Nilai UTS
- Kecepatan belajar
- Hasil clustering
- Mastery score
- Learning gap
- Prioritas pembelajaran
- Area penilaian dengan nilai terendah

---

## 2. System Architecture

Alur utama sistem:

```text
Student Input
     |
     v
FastAPI Request Validation
     |
     v
Saved Scaler + K-Means Model
     |
     v
Student Prediction
     |
     +--> Average Score
     |
     +--> Mastery Score
     |
     +--> Learning Gap
     |
     +--> Learning Priority
     |
     v
Dynamic Recommendation Engine
     |
     +--> Weakest Assessment
     +--> Difficulty Level
     +--> Learning Strategy
     +--> Recommendation Focus
     |
     v
Gemini Generative AI
     |
     v
Personalized Lesson Plan
     |
     v
JSON API Response
```

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas |
| Numerical Processing | NumPy / Scikit-learn dependencies |
| Machine Learning | Scikit-learn |
| Clustering | K-Means |
| Feature Scaling | StandardScaler |
| Model Serialization | Joblib |
| Backend API | FastAPI |
| API Server | Uvicorn |
| Data Validation | Pydantic |
| Generative AI | Google Gemini API |
| Gemini SDK | `google-genai` |
| Environment Configuration | `python-dotenv` |
| Output Format | CSV, JSON, JSONL |

---

## 4. Project Structure

Struktur proyek utama:

```text
ml-joints-ugm/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── students_clustered.csv
│       ├── cluster_centroids.csv
│       ├── student_profiles.csv
│       ├── learning_recommendations.csv
│       ├── lesson_plan_prompts.jsonl
│       ├── lesson_plans.jsonl
│       └── lesson_plan_errors.jsonl
│
├── models/
│   ├── scaler.pkl
│   ├── kmeans.pkl
│   └── cluster_metadata.json
│
├── src/
│   ├── data/
│   │   ├── generate_data.py
│   │   └── explore_data.py
│   │
│   ├── preprocessing/
│   │   └── features.py
│   │
│   ├── models/
│   │   ├── clustering.py
│   │   └── student_profile.py
│   │
│   ├── recommendation/
│   │   ├── learning_recommendation.py
│   │   └── dynamic_recommendation.py
│   │
│   ├── llm/
│   │   ├── prompt_builder.py
│   │   ├── gemini_lesson_plan.py
│   │   ├── retry_failed_lesson_plan.py
│   │   └── dynamic_lesson_plan.py
│   │
│   ├── validation/
│   │   └── validate_lesson_plans.py
│   │
│   └── api/
│       ├── main.py
│       └── ml_predictor.py
│
├── .env
├── requirements.txt
└── README.md
```

Nama file atau folder dapat berkembang selama pengembangan proyek.

---

## 5. Dataset

Pada tahap awal, proyek menggunakan data siswa sintetis untuk menguji alur machine learning dan rekomendasi.

Fitur utama yang digunakan dalam model clustering:

| Feature | Description | Type |
|---|---|---|
| `diagnostic_score` | Nilai tes diagnostik awal | Numeric |
| `assignment_avg` | Rata-rata nilai tugas | Numeric |
| `quiz_avg` | Rata-rata nilai kuis | Numeric |
| `uts_score` | Nilai UTS | Numeric |
| `learning_speed` | Indikator kecepatan belajar | Numeric |

Fitur berikut juga digunakan dalam data pendukung atau proses pembuatan lesson plan:

- `student_id`
- `visual_preference`
- `audio_preference`
- `interest`
- `observation_note`

> Tidak semua fitur digunakan sebagai input clustering. Model clustering saat ini menggunakan lima fitur numerik yang telah ditentukan dalam preprocessing.

---

## 6. Data Preprocessing

Preprocessing dilakukan sebelum data masuk ke algoritma clustering.

Fitur numerik yang digunakan:

```python
FEATURES = [
    "diagnostic_score",
    "assignment_avg",
    "quiz_avg",
    "uts_score",
    "learning_speed",
]
```

Fitur tersebut memiliki skala yang berbeda. Contohnya, nilai akademik berada pada rentang 0–100, sedangkan `learning_speed` berada pada rentang 0–1.

Untuk mengurangi pengaruh perbedaan skala, digunakan `StandardScaler`.

Secara umum, StandardScaler melakukan standardisasi dengan rumus:

```text
z = (x - mean) / standard_deviation
```

Scaler yang telah dilatih disimpan agar data siswa baru diproses menggunakan transformasi yang sama dengan data training.

File model:

```text
models/scaler.pkl
```

### Alasan Menyimpan Scaler

Scaler harus menggunakan parameter yang sama ketika melakukan prediksi terhadap siswa baru. Jika data baru diskalakan ulang menggunakan parameter yang berbeda, hasil prediksi clustering dapat menjadi tidak konsisten.

---

## 7. Machine Learning: K-Means Clustering

### 7.1 Konsep

EduAdapt AI menggunakan algoritma **K-Means Clustering**, yaitu algoritma unsupervised learning yang mengelompokkan data berdasarkan kedekatan karakteristiknya.

Model saat ini menggunakan:

```python
KMeans(
    n_clusters=3,
    n_init=10,
    random_state=42
)
```

Tiga kelompok tersebut diberi label berdasarkan karakteristik centroid:

- `Fast Learner`
- `Steady Learner`
- `Needs Guidance`

Label tersebut merupakan interpretasi dari hasil clustering, bukan label yang diberikan langsung oleh dataset.

### 7.2 Mengapa K-Means?

K-Means digunakan karena:

- Relatif sederhana untuk diterapkan.
- Cocok untuk eksplorasi pola kelompok siswa.
- Dapat digunakan tanpa label kelas yang sudah tersedia.
- Hasilnya dapat digunakan sebagai salah satu komponen student profiling.

Namun, K-Means tidak secara otomatis memahami kondisi psikologis atau kemampuan siswa. Model hanya mengelompokkan siswa berdasarkan fitur yang diberikan.

### 7.3 Hasil Clustering

Pada salah satu proses eksperimen dengan 50 data siswa:

| Cluster Interpretation | Jumlah Siswa |
|---|---:|
| Needs Guidance | 21 |
| Steady Learner | 16 |
| Fast Learner | 13 |

Silhouette score yang diperoleh:

```text
0.1758
```

Silhouette score tersebut menunjukkan bahwa pemisahan cluster masih relatif lemah. Oleh karena itu, hasil clustering sebaiknya diperlakukan sebagai indikasi awal untuk membantu rekomendasi, bukan sebagai kebenaran mutlak mengenai kemampuan siswa.

Hasil clustering disimpan dalam:

```text
data/processed/students_clustered.csv
data/processed/cluster_centroids.csv
```

Model dan metadata disimpan dalam:

```text
models/kmeans.pkl
models/cluster_metadata.json
```

---

## 8. Model Serialization

Agar model yang sudah dilatih dapat digunakan oleh API, beberapa komponen disimpan ke dalam folder `models`.

| File | Function |
|---|---|
| `scaler.pkl` | Menyimpan StandardScaler |
| `kmeans.pkl` | Menyimpan model K-Means |
| `cluster_metadata.json` | Menyimpan konfigurasi dan mapping cluster |

Pada saat API menerima data siswa baru:

1. Data input disusun sesuai urutan fitur training.
2. Data diproses menggunakan `scaler.pkl`.
3. Data hasil scaling dikirim ke `kmeans.pkl`.
4. Model menghasilkan `cluster_id`.
5. `cluster_metadata.json` digunakan untuk mendapatkan nama cluster.

Pendekatan ini memastikan API tidak melatih ulang model setiap kali menerima request.

---

## 9. Student Profiling

Setelah proses clustering, sistem membentuk profil belajar siswa.

### 9.1 Average Score

Average score dihitung dari empat komponen nilai:

```text
diagnostic_score
assignment_avg
quiz_avg
uts_score
```

Secara umum:

```text
average_score =
    (diagnostic_score
    + assignment_avg
    + quiz_avg
    + uts_score) / 4
```

### 9.2 Mastery Score

Mastery score dihitung dengan membagi average score dengan 100:

```text
mastery_score = average_score / 100
```

Nilai mastery score berada pada rentang 0–1.

Contoh:

```text
average_score = 65.75

mastery_score = 65.75 / 100
              = 0.6575
```

### 9.3 Learning Gap

Learning gap dihitung dengan:

```text
learning_gap = 1 - mastery_score
```

Contoh:

```text
learning_gap = 1 - 0.6575
             = 0.3425
```

Learning gap menunjukkan jarak antara mastery score siswa dan nilai maksimum yang digunakan oleh perhitungan ini.

### 9.4 Learning Priority

Prioritas pembelajaran ditentukan menggunakan aturan:

```text
Jika learning_gap >= 0.40:
    High

Jika learning_gap >= 0.25:
    Medium

Selain itu:
    Low
```

Prioritas ini merupakan aturan berbasis threshold dan bukan hasil klasifikasi supervised learning.

---

## 10. Rule-Based Learning Recommendation

Sistem rekomendasi dinamis menentukan area penilaian dengan nilai paling rendah dari:

```text
Diagnostic
Assignment
Quiz
UTS
```

Contoh input:

```json
{
  "diagnostic_score": 70,
  "assignment_avg": 75,
  "quiz_avg": 60,
  "uts_score": 58
}
```

Area terlemah:

```text
UTS = 58
```

### 10.1 Difficulty Mapping

Tingkat kesulitan ditentukan berdasarkan nilai terendah:

| Nilai Terendah | Difficulty | Strategy |
|---|---|---|
| < 60 | Dasar | Pembelajaran konsep dan latihan bertahap |
| 60–74.99 | Menengah | Latihan terarah dan penguatan konsep |
| >= 75 | Lanjutan | Latihan penerapan dan soal tantangan |

### 10.2 Focus Berdasarkan Priority

| Priority | Focus |
|---|---|
| High | Penguatan konsep dasar |
| Medium | Penguatan materi yang masih lemah |
| Low | Pengembangan kemampuan lanjutan |

Hasil rekomendasi berisi:

- `focus`
- `weakest_assessment`
- `weakest_score`
- `difficulty`
- `learning_strategy`
- `reason`

Pendekatan ini bersifat explainable karena alasan rekomendasi dapat ditelusuri dari nilai yang digunakan.

---

## 11. Generative AI: Gemini Lesson Plan

Setelah sistem menghasilkan prediksi dan rekomendasi, data tersebut dikirim ke Gemini untuk menghasilkan lesson plan personal.

### Input Gemini

Profil yang diberikan kepada Gemini meliputi:

- Student ID
- Nilai diagnostik
- Rata-rata tugas
- Rata-rata kuis
- Nilai UTS
- Learning speed
- Cluster name
- Mastery score
- Learning gap
- Priority
- Focus rekomendasi
- Area penilaian terlemah
- Difficulty
- Learning strategy
- Reason

### Konteks Pembelajaran

Saat ini prompt menggunakan konteks:

```text
Mata pelajaran: Matematika
Materi: Persamaan Linear Satu Variabel
Tingkat: SMP
Bahasa: Indonesia
```

### Struktur Output

Gemini diarahkan untuk menghasilkan JSON dengan struktur:

```json
{
  "student_id": "S051",
  "learning_objective": [],
  "lesson_explanation": [],
  "learning_activities": [],
  "practice_questions": [
    {
      "question": "...",
      "difficulty": "Easy",
      "answer": "..."
    }
  ],
  "evaluation_method": []
}
```

### Validasi Output

Respons Gemini divalidasi untuk memastikan:

- Response berbentuk object JSON.
- Semua field wajib tersedia.
- `student_id` sesuai dengan siswa yang diminta.
- `practice_questions` berbentuk list.
- Setiap latihan memiliki `question`, `difficulty`, dan `answer`.
- Nilai difficulty termasuk `Easy`, `Medium`, atau `Hard`.

### Catatan Rate Limit dan Timeout

Pemanggilan Gemini dapat mengalami error seperti:

- `429 RESOURCE_EXHAUSTED`
- `503 UNAVAILABLE`
- `504 DEADLINE_EXCEEDED`

Implementasi telah menyediakan mekanisme retry untuk beberapa error sementara. Namun, retry tidak menjamin seluruh request selalu berhasil karena keterbatasan jaringan, waktu respons model, dan kuota API.

---

## 12. Backend API

Backend dibangun menggunakan FastAPI.

Jalankan API dengan:

```bash
python -m uvicorn src.api.main:app --reload
```

Dokumentasi Swagger tersedia di:

```text
http://127.0.0.1:8000/docs
```

OpenAPI JSON tersedia di:

```text
http://127.0.0.1:8000/openapi.json
```

### 12.1 Health Check

```http
GET /health
```

Digunakan untuk memeriksa apakah API aktif.

### 12.2 Get All Students

```http
GET /students
```

Mengambil data student profile yang tersedia pada file hasil preprocessing.

### 12.3 Get Student Profile

```http
GET /students/{student_id}
```

Mengambil profil siswa berdasarkan ID.

Contoh:

```http
GET /students/S001
```

### 12.4 Get Student Lesson Plan

```http
GET /students/{student_id}/lesson-plan
```

Mengambil lesson plan yang telah tersedia pada file `lesson_plans.jsonl`.

### 12.5 Get Student Dashboard Data

```http
GET /students/{student_id}/dashboard
```

Menggabungkan data yang diperlukan untuk kebutuhan dashboard.

### 12.6 Analyze New Student

```http
POST /students/analyze
```

Endpoint ini menerima data siswa baru, kemudian melakukan:

1. Prediksi cluster.
2. Perhitungan mastery score.
3. Perhitungan learning gap.
4. Penentuan priority.
5. Pembuatan rekomendasi dinamis.
6. Pembuatan lesson plan melalui Gemini.

#### Request Body

```json
{
  "student_id": "S051",
  "diagnostic_score": 70,
  "assignment_avg": 75,
  "quiz_avg": 60,
  "uts_score": 58,
  "learning_speed": 0.5
}
```

#### Validasi Input

Field nilai akademik harus berada pada rentang:

```text
0 sampai 100
```

Field `learning_speed` harus berada pada rentang:

```text
0 sampai 1
```

`student_id` harus berupa string dengan panjang 1 sampai 50 karakter.

#### Contoh Response

```json
{
  "student_id": "S051",
  "prediction": {
    "cluster_id": 1,
    "cluster_name": "Needs Guidance",
    "average_score": 65.75,
    "mastery_score": 0.6575,
    "learning_gap": 0.3425,
    "priority": "Medium"
  },
  "recommendation": {
    "focus": "Penguatan materi yang masih lemah",
    "weakest_assessment": "UTS",
    "weakest_score": 58,
    "difficulty": "Dasar",
    "learning_strategy": "Pembelajaran konsep dan latihan bertahap",
    "reason": "Nilai terendah terdapat pada UTS dengan nilai 58.00."
  },
  "lesson_plan": {
    "student_id": "S051",
    "learning_objective": [],
    "lesson_explanation": [],
    "learning_activities": [],
    "practice_questions": [],
    "evaluation_method": []
  }
}
```

Isi array pada `lesson_plan` bergantung pada respons Gemini.

---

## 13. Installation

### 13.1 Clone Repository

```bash
git clone <repository-url>
cd ml-joints-ugm
```

Ganti `<repository-url>` dengan URL repository yang sebenarnya.

### 13.2 Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Aktivasi:

```powershell
.venv\Scripts\activate
```

### 13.3 Install Dependencies

```bash
pip install -r requirements.txt
```

Jika file `requirements.txt` belum diperbarui, pastikan dependensi utama tersedia:

```bash
pip install pandas scikit-learn joblib fastapi uvicorn python-dotenv google-genai pydantic
```

### 13.4 Environment Variables

Buat file `.env` pada root project:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_gemini_model
```

Jangan memasukkan API key ke dalam repository publik.

Tambahkan `.env` ke `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
*.pyc
```

---

## 14. Running the Machine Learning Pipeline

Urutan proses yang direkomendasikan:

### Step 1 — Generate Data

```bash
python -m src.data.generate_data
```

### Step 2 — Explore Data

```bash
python -m src.data.explore_data
```

### Step 3 — Train Clustering Model

```bash
python -m src.models.clustering
```

Proses ini menghasilkan:

```text
models/scaler.pkl
models/kmeans.pkl
models/cluster_metadata.json
```

### Step 4 — Create Student Profiles

```bash
python -m src.models.student_profile
```

### Step 5 — Generate Learning Recommendations

```bash
python -m src.recommendation.learning_recommendation
```

### Step 6 — Build Prompts

```bash
python -m src.llm.prompt_builder
```

### Step 7 — Generate Lesson Plans

```bash
python -m src.llm.gemini_lesson_plan
```

### Step 8 — Validate Lesson Plans

```bash
python -m src.validation.validate_lesson_plans
```

Contoh hasil validasi yang pernah diperoleh:

```text
Total record: 50
Jumlah student ID unik: 50
Record valid: 50
Record tidak valid: 0
STATUS: VALID
```

### Step 9 — Run API

```bash
python -m uvicorn src.api.main:app --reload
```

---

## 15. Model Evaluation

Evaluasi yang digunakan dalam tahap clustering adalah **Silhouette Score**.

Silhouette score membantu mengukur seberapa dekat sebuah data dengan anggota cluster-nya sendiri dibandingkan dengan cluster lainnya.

Nilai yang lebih tinggi biasanya menunjukkan pemisahan cluster yang lebih baik. Namun, interpretasi harus mempertimbangkan:

- Jumlah data.
- Jumlah cluster.
- Distribusi fitur.
- Kualitas fitur.
- Karakteristik dataset.
- Tujuan penggunaan model.

Pada eksperimen awal, silhouette score tercatat:

```text
0.1758
```

Nilai tersebut menunjukkan bahwa struktur cluster belum kuat. Oleh sebab itu, model sebaiknya tidak digunakan sebagai satu-satunya dasar untuk mengambil keputusan pendidikan.

### Keterbatasan Evaluasi Saat Ini

Evaluasi yang tersedia masih berfokus pada clustering dan validasi struktur output lesson plan. Belum tersedia evaluasi komprehensif terhadap:

- Akurasi peningkatan hasil belajar siswa.
- Efektivitas rekomendasi.
- Kesesuaian lesson plan menurut guru.
- Perbandingan dengan baseline rekomendasi lain.
- Evaluasi pedagogis dengan pengguna nyata.

---

## 16. Design Considerations

### 16.1 Clustering Bukan Diagnosis Kemampuan Mutlak

Label seperti `Fast Learner`, `Steady Learner`, dan `Needs Guidance` merupakan interpretasi berdasarkan fitur yang tersedia.

Label tersebut tidak boleh dianggap sebagai diagnosis permanen atau penilaian menyeluruh terhadap seorang siswa.

### 16.2 Synthetic Data

Jika dataset yang digunakan merupakan data sintetis, hasil model belum dapat dianggap mewakili populasi siswa nyata.

Data nyata membutuhkan:

- Persetujuan penggunaan data.
- Perlindungan privasi.
- Pemeriksaan kualitas data.
- Penanganan missing value.
- Evaluasi bias.
- Validasi bersama tenaga pendidik.

### 16.3 Rule-Based Recommendation

Sistem rekomendasi saat ini menggunakan aturan yang telah ditentukan. Sistem belum belajar secara langsung dari hasil keberhasilan atau kegagalan rekomendasi sebelumnya.

### 16.4 Generative AI Output

Lesson plan dari Gemini perlu divalidasi sebelum digunakan dalam konteks pendidikan nyata. Validasi JSON memastikan struktur output, tetapi tidak otomatis menjamin:

- Ketepatan pedagogis.
- Kebenaran setiap penjelasan.
- Kesesuaian dengan kurikulum.
- Tingkat kesulitan yang sempurna.
- Tidak adanya kesalahan pada jawaban soal.

---

## 17. Current Limitations

Fitur yang belum menjadi bagian dari implementasi saat ini:

- Penyimpanan siswa baru ke database.
- PostgreSQL atau database relasional.
- Authentication dan authorization.
- Role-based access untuk siswa, guru, dan administrator.
- Frontend dashboard produksi.
- Monitoring model.
- Model retraining otomatis.
- Evaluasi dengan dataset siswa nyata.
- Feedback loop dari hasil belajar siswa.
- Versioning dataset dan model secara formal.

---

## 18. Future Development

Pengembangan berikutnya dapat mencakup:

### Backend and Database

- Integrasi PostgreSQL.
- Penyimpanan profil siswa.
- Penyimpanan hasil analisis.
- Penyimpanan lesson plan.
- Endpoint untuk riwayat pembelajaran.
- Authentication dan authorization.

### Machine Learning

- Eksperimen dengan jumlah cluster berbeda.
- Perbandingan K-Means dengan metode clustering lain.
- Feature engineering.
- Evaluasi clustering yang lebih komprehensif.
- Model supervised learning jika label hasil belajar tersedia.
- Model retraining dengan data baru.

### Recommendation System

- Rekomendasi berdasarkan riwayat aktivitas.
- Rekomendasi berdasarkan hasil latihan.
- Feedback dari hasil evaluasi siswa.
- Pembaruan tingkat kesulitan secara adaptif.
- Perbandingan rule-based dengan recommendation model.

### Generative AI

- Validasi fakta dan jawaban secara otomatis.
- Penggunaan template lesson plan yang lebih terstruktur.
- Penyimpanan prompt dan respons untuk audit.
- Pengendalian biaya dan rate limit.
- Evaluasi kualitas lesson plan oleh guru.
- Dukungan terhadap materi dan jenjang pendidikan lain.

---

## 19. Git Workflow

Contoh commit yang digunakan selama pengembangan:

```bash
git add src/models/
git commit -m "feat: add clustering model persistence"
```

```bash
git add src/api/
git commit -m "feat: add student analysis API"
```

```bash
git add src/recommendation/ src/api/main.py
git commit -m "feat: add dynamic learning recommendation"
```

```bash
git add src/llm/ src/api/main.py
git commit -m "feat: integrate dynamic Gemini lesson plan"
```

Gunakan commit yang singkat dan menjelaskan perubahan utama.

---

## 20. Example Workflow

Contoh data siswa:

```json
{
  "student_id": "S051",
  "diagnostic_score": 70,
  "assignment_avg": 75,
  "quiz_avg": 60,
  "uts_score": 58,
  "learning_speed": 0.5
}
```

Sistem kemudian:

1. Menggunakan scaler yang telah disimpan.
2. Memprediksi cluster siswa.
3. Menghitung average score sebesar `65.75`.
4. Menghitung mastery score sebesar `0.6575`.
5. Menghitung learning gap sebesar `0.3425`.
6. Menentukan priority `Medium`.
7. Menentukan UTS sebagai area dengan nilai terendah.
8. Menentukan difficulty `Dasar`.
9. Membentuk strategi pembelajaran bertahap.
10. Mengirim data ke Gemini.
11. Menghasilkan lesson plan dalam format JSON.

---

## 21. Disclaimer

EduAdapt AI merupakan proyek pengembangan dan eksperimen teknologi pembelajaran. Hasil clustering, rekomendasi, dan lesson plan harus digunakan sebagai alat bantu, bukan sebagai pengganti penilaian profesional guru atau evaluasi pembelajaran yang menyeluruh.

Penggunaan pada data siswa nyata perlu memperhatikan privasi, keamanan data, persetujuan, transparansi, dan evaluasi dampak terhadap siswa.

---

## 22. Author

Developed by **Muhammad Rizal Ramzi**

Focus:

- Full Stack Development
- Machine Learning
- Data Science
- Generative AI
- Educational Technology
