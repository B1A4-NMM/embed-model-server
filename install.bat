@echo off
REM Windows용 설치 스크립트

echo === Windows 설치 스크립트 시작 ===

REM Python 설치 확인
python --version
if errorlevel 1 (
    echo Python이 설치되어 있지 않습니다. Python 3.9+를 설치해주세요.
    pause
    exit /b 1
)

REM 가상환경 생성
echo 가상환경 생성 중...
python -m venv venv

REM 가상환경 활성화
echo 가상환경 활성화 중...
call venv\Scripts\activate.bat

REM pip 업그레이드
echo pip 업그레이드 중...
python -m pip install --upgrade pip

REM requirements.txt 설치
echo 라이브러리 설치 중...
pip install -r requirements.txt

echo === 설치 완료 ===
echo 가상환경 활성화: venv\Scripts\activate.bat
echo e5.py 실행: python e5.py
echo reranker.py 실행: python reranker.py
pause 