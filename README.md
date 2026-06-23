# IDS Evaluation — Comparative Feature Selection (UNSW-NB15)

Aplikasi Streamlit untuk membandingkan metode *feature selection* (Filter
SelectKBest χ² vs Wrapper RFE) terhadap baseline *all-features*, menggunakan
Random Forest pada dataset UNSW-NB15.

Dataset training & testing sudah dibundel di folder `data/`, jadi begitu
aplikasi ini di-deploy, siapa pun yang membuka link-nya bisa langsung melihat
hasil analisis tanpa perlu upload file apa pun. Mereka tetap bisa upload CSV
sendiri di sidebar untuk mengganti dataset.

## Struktur folder

```
ids-app/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── data/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
└── README.md
```

## Cara Deploy ke Streamlit Community Cloud (gratis)

### 1. Push folder ini ke GitHub

```bash
cd ids-app
git init
git add .
git commit -m "Initial commit: IDS feature selection app"
git branch -M main
git remote add origin https://github.com/<username-kamu>/<nama-repo>.git
git push -u origin main
```

> Buat repo baru dulu di GitHub (boleh public/private) sebelum `git push`.
> Ukuran `data/` sekitar 46 MB — masih aman untuk `git push` biasa (limit
> GitHub per file 100 MB).

### 2. Deploy

- Buka https://share.streamlit.io (login dengan akun GitHub)
- Klik **"New app"**
- Pilih repo, branch `main`, dan file utama `app.py`
- Klik **Deploy**

Tunggu beberapa menit sampai build selesai. Kamu akan mendapat URL publik
seperti:

```
https://<nama-app-kamu>.streamlit.app
```

Link ini bisa langsung dibagikan ke siapa saja — mereka tidak perlu install
apa pun, cukup buka link di browser.

### 3. Update aplikasi di kemudian hari

Setiap kali kamu `git push` perubahan ke branch `main`, Streamlit Cloud akan
otomatis rebuild dan update aplikasi yang sudah live.

## Menjalankan secara lokal (opsional, untuk testing sebelum deploy)

```bash
pip install -r requirements.txt
streamlit run app.py
```
