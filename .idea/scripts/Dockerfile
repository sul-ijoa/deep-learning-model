# 공식 Python 런타임 이미지를 부모 이미지로 사용
FROM python:3.11-slim

# 컨테이너 내에서 작업 디렉터리 설정
WORKDIR /app

# 현재 디렉토리의 내용을 컨테이너 내의 /requirements.txt로 복사
COPY requirements.txt /requirements.txt

# requirements.txt에 지정된 필요한 패키지 설치
RUN pip install -r /requirements.txt
RUN pip list

# 나머지 애플리케이션 코드를 작업 디렉토리로 복사
COPY . /app

# 포트 5000을 외부에 노출
EXPOSE 5000

# 컨테이너 시작 시 실행할 명령어 지정
CMD ["python", "app.py"]