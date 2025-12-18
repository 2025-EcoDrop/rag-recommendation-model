# 💊 rag-recommendation-model
LLM과 RAG(Retrieval-Augmented Generation)를 활용한 증상 기반 약품 추천 API입니다. </br>
FastAPI로 서버를 구성하고, 벡터 검색 결과를 LLM에 제공하여 추천 결과를 생성합니다. </br>
</br>

## ⚠️ 참고 사항
- 본 API는 의료 진단을 대체하지 않습니다.  </br>
- LLM 호출로 인해 응답 시간이 다소 길 수 있습니다.  </br>
</br>


## 📁 주요 파일 구조
```bash
rag-recommendation-model
├── app/
│   ├── database/        # 데이터 베이스와 연결
│   ├── main.py          # FastAPI 앱 진입점
│   ├── schemas.py       # Request / Response 모델
│   ├── rag.py             # 임베딩 및 벡터 검색
│   ├── llm_model.py     # LLM 응답 생성 로직
│   └── routes/
│       └── chat.py      # /chat API
├── requirements.txt
└── README.md
```

## ▶️ 실행
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

</br>
