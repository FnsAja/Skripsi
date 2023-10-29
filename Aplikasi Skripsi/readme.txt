Cara menjalankan program:
1. Masuk ke "App/.env/pyvenv.cfg" dan ubah configurasi sesuai dengan device masing-masing
2. Buka terminal dan arahkan ke directory "./App" lalu jalankan command ".\.env\Scripts\activate" untuk mengaktifkan virtual environtment
3. Lalu jalankan command "cd .\Flask\"
4. Lalu jalankan "flask --app main run --debug"
5. Lalu ketikan url berikut pada browser "127.0.0.1:5000"

Program ini dibuat menggunakan python dengan framework "Flask", yang berupa website dengan beberapa fitur machine learning didalamnya.
Algoritma yang digunakan adalah Support Vector Machine dengan metode validasi menggunakan 10 Fold Cross Validation dan Confusion Matrix.
