@echo off
chcp 65001 > nul
title Пепе Ассистент - Запуск
echo ========================================
echo    Запуск ассистента Пепе
echo ========================================
echo.

cd /d "C:\Users\dekan\Desktop\Pepe_VA"

echo Активация виртуального окружения...
call .\venv\Scripts\activate.bat

echo Проверка установки необходимых пакетов...
pip install -r requirements.txt

echo Запуск ассистента...
python Pepe.py

echo.
echo Ассистент завершил работу.
echo Нажмите любую клавишу для выхода...
pause > nul