#!/bin/bash
# EC2 인스턴스에서 실행할 설치 스크립트

echo "=== EC2 설치 스크립트 시작 ==="

# 시스템 업데이트
echo "시스템 업데이트 중..."
sudo yum update -y

# Python 3.9+ 설치 (Amazon Linux 2)
echo "Python 설치 중..."
sudo yum install python3 python3-pip -y

# 가상환경 생성
echo "가상환경 생성 중..."
python3 -m venv venv
source venv/bin/activate

# pip 업그레이드
echo "pip 업그레이드 중..."
pip install --upgrade pip

# requirements.txt 설치
echo "라이브러리 설치 중..."
pip install -r requirements.txt

# 포트 열기 (보안 그룹에서도 설정 필요)
echo "웹 서버 설정 중..."
sudo yum install -y httpd
sudo systemctl start httpd
sudo systemctl enable httpd

echo "=== 설치 완료 ==="
echo "가상환경 활성화: source venv/bin/activate"
echo "e5.py 실행: python e5.py"
echo "reranker.py 실행: python reranker.py" 