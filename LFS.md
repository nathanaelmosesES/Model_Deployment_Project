# Git LFS — Setup & Info

## Apakah gratis?

**Ya, gratis** dengan batas berikut (GitHub Free):

| Kuota | Limit |
|-------|-------|
| Storage | 1 GB |
| Bandwidth (download/bulan) | 1 GB |

Model kita (`model.joblib`) berukuran ~191 MB — masih dalam batas gratis.
Jika melebihi kuota, GitHub menawarkan Data Pack seharga **$5/bulan** (50 GB storage + 50 GB bandwidth).

---

## Langkah-langkah setup (yang sudah dilakukan)

### 1. Hapus file besar dari git index (tanpa hapus file fisik)
```bash
git rm --cached artifacts/model.joblib
```

### 2. Install dan aktifkan Git LFS
```bash
git lfs install
```

### 3. Daftarkan file yang akan di-track LFS
```bash
git lfs track "artifacts/model.joblib"
```
Perintah ini membuat/mengupdate file `.gitattributes` di root project.

### 4. Stage `.gitattributes` dan file model
```bash
git add .gitattributes artifacts/model.joblib
```

### 5. Amend commit (karena ini commit pertama, tidak bisa reset)
```bash
git commit --amend --no-edit
```

### 6. Push ke GitHub (LFS upload otomatis)
```bash
git push -u origin main
```

---

## Cara kerja LFS

```
Repo GitHub (git)          GitHub LFS Storage
┌─────────────────┐        ┌──────────────────┐
│ model.joblib    │──────▶ │  file asli 191MB │
│ (pointer 134B)  │        │                  │
└─────────────────┘        └──────────────────┘
```

Git hanya menyimpan **pointer kecil** (~134 byte) di repo. File aslinya tersimpan terpisah di LFS storage. Saat `git clone` atau Streamlit Cloud deploy, file asli otomatis didownload.

---

## Jika perlu retrain dan update model

```bash
# Setelah training ulang, cukup:
git add artifacts/model.joblib
git commit -m "update: retrain model"
git push
```

LFS menangani upload otomatis — tidak perlu langkah tambahan.

---

## Cek status LFS

```bash
git lfs ls-files        # lihat file yang di-track LFS
git lfs status          # cek status file LFS
```
