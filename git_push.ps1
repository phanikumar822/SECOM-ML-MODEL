git init
Out-File -FilePath .gitignore -InputObject "__pycache__/" -Encoding UTF8 -Append
git add .
git commit -m "Initial commit: SECOM ML Model Pipeline"
git branch -M main
git remote remove origin 2>$null
git remote add origin https://github.com/phanikumar822/SECOM-ML-MODEL.git
git push -u origin main
