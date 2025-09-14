# Dev Assignment MCP (fastmcp + PostgreSQL + (옵션) OpenAI)

개발자 업무를 자동으로 배분하는 **MCP 서버**입니다.  
- 표준 **MCP(stdio)** 방식으로 LLM 클라이언트(Cursor/VS Code/Claude Desktop 등)와 연동
- 규칙 기반(유사도/업무량/기술 매칭/프로젝트 보너스) + (옵션) **LLM 보정**
- 개발자는 내 업무 조회 + 과거 유사 TASK 추천
- 완료 시 처리내용/방법/관련 DB 기록 → **다음 유사 TASK 추천에 활용**

---

## 기능(요약)

### 관리자
- `admin_add_developer(project, name, experience_years, skills[], work_experiences, workload=0)`
- `admin_add_task(title, requirements, required_skills[], project=None)` → **자동 배정 + 연관 과거 TASK 추천**
- `admin_set_weights(...)` → 자동 분배 가중치 조정
- `seed_demo_data()` → 샘플 데이터 삽입

### 개발자
- `dev_my_tasks(developer_name)` → 내 업무 + 연관 과거 TASK 리스트
- `dev_complete_task(task_id, resolution_notes, resolution_method, related_db)` → 완료 + 기록

### 검색
- `search_similar_tasks(query, top_k=5)` → 자연어로 과거 유사 TASK 탐색

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
$ uv venv --python 3.13
$ uv init
$ uv add fastmcp psycopg2-binary openai
```

## Postgres 확장 (서버에서 1회)
```
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

## VS Code / Cursor 연결 (.cursor/mcp.json)
프로젝트 루트에 .cursor/mcp.json 생성:
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
VS Code(또는 Cursor/Claude Desktop) 재시작 → 채팅창에서 MCP 함수 사용 가능.
