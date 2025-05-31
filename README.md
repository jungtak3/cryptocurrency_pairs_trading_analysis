# 암호화폐 페어 트레이딩 분석

Saji T.G. (2021) 논문 기반 암호화폐 페어 트레이딩 전략 구현체임.

## 설치

```bash
pip install -r requirements.txt
```

## 데이터 준비

`data/` 폴더에 CSV 파일들 넣음:
- `BTCUSDT_20170817_20250430.csv`
- `ETHUSDT_20170817_20250430.csv` 
- `LTCUSDT_20171213_20250430.csv`
- `NEOUSDT_20171120_20250430.csv`

## 실행

### 기본 분석
```bash
python examples/generate_reports.py
```

### 커스텀 날짜 분석
```bash
# 날짜 계산기 (예: 2025-04-30 종료일로 6개월 간격 패널 생성)
python examples/date_calculator.py 20250430

# 커스텀 날짜로 분석 실행
python examples/analyze_custom_dates.py 20250430
```

## 출력

6개 테이블 생성됨:
1. 기초통계량
2. 상관관계/거리
3. 가격 ADF 검정
4. 공적분 분석
5. 잔차 ADF 검정
6. 수익성 비교

마크다운 리포트가 `reports/` 폴더에 저장됨.


## 구조

```
├── src/                    # 핵심 코드
├── examples/              # 실행 스크립트
├── data/                  # 데이터 폴더
├── reports/               # 리포트 출력
└── docs/                  # 문서