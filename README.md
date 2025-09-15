# Dev Assignment MCP (fastmcp + PostgreSQL + (옵션) OpenAI)

개발자 업무를 자동으로 배분하는 **MCP 서버**입니다.  
- 표준 **MCP(stdio)** 방식으로 LLM 클라이언트(Cursor/VS Code/Claude Desktop 등)와 연동
- 규칙 기반(유사도/업무량/기술 매칭/프로젝트 보너스) + (옵션) **LLM 보정**
- 개발자는 내 업무 조회 + 과거 유사 TASK 추천
- 완료 시 처리내용/방법/관련 DB 기록 → **다음 유사 TASK 추천에 활용**

---

## 기능(요약)

1. 개발자 관리  
admin_add_developer - 개발자 등록  
get_all_developers - 전체 개발자 목록 조회  
  
2. 업무 관리  
admin_add_task - 업무 등록 (자동 배정)  
get_all_tasks - 전체 업무 목록 조회  
dev_my_tasks - 개발자별 업무 조회  
dev_complete_task - 업무 완료 처리  
search_similar_tasks - 유사 업무 검색  
  
3. 시스템 관리  
seed_demo_data - 데모 데이터 초기화  
admin_set_weights - 배정 알고리즘 가중치 설정  

---

## 분배 규칙

1) **과거 유사 TASK 경험(최대 유사도, 유사 사례 수)**  
2) **업무량이 적은 순** (편차가 크면 저업무량 개발자 가중)  
3) **필요 기술 ∩ 보유 기술** 개수  
4) **프로젝트 일치 보너스**  
5) (옵션) **LLM 적합도 보정** (규칙 점수 + LLM 점수 혼합)

유사도는 기본적으로 **Postgres `pg_trgm`**(trigram similarity) 사용.  
(옵션) OpenAI 임베딩/랭킹 보정 사용 가능.

---

## 환경변수

필수(PostgreSQL):  
PGHOST=localhost  
PGPORT=5432  
PGDATABASE=mcpdb  
PGUSER=postgres  
PGPASSWORD=postgres  

선택(OpenAI 연동):  
OPENAI_API_KEY=sk-...  
OPENAI_MODEL=gpt-5 # 또는 gpt-4o 등 사용 가능한 최신 모델명  
OPENAI_EMBED_MODEL=text-embedding-3-large  


## 가상환경 설정
```
$ uv python install 3.13   
$ uv init
$ uv venv --python 3.13.7
$ source .venv/bin/activate  
$ uv add fastmcp psycopg2-binary openai
```

## Postgres 확장 (서버에서 1회)
```
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## VS Code / Cursor 연결 (.cursor/mcp.json)
Cursor Setting -> MCP & Integrations > MCP Tools 활성화  
```
{
  "mcpServers": {
    "dev-assignment-mcp": {
      "command": "python",
      "args": ["server.py"],
      "cwd": "/절대/혹은/프로젝트/경로",
      "env": {
        "PGHOST": "localhost",
        "PGPORT": "5432",
        "PGDATABASE": "mcpdb",
        "PGUSER": "postgres",
        "PGPASSWORD": "postgres",
        "OPENAI_API_KEY": "sk-... (옵션)",
        "OPENAI_MODEL": "gpt-5",
        "OPENAI_EMBED_MODEL": "text-embedding-3-large"
      }
    }
  }
}

```

<img width="481" height="489" alt="image" src="https://github.com/user-attachments/assets/854edab1-ed24-4d5a-b9f7-2af2570d852b" />

