
import os
import json
import math
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any

from fastmcp import FastMCP

# ========== (옵션) OpenAI LLM / 임베딩 ==========
USE_LLM = bool(os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4oll")  # 환경에 맞는 최신 모델명 사용
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

if USE_LLM:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    except Exception as e:
        print("[WARN] OpenAI init failed:", e)
        USE_LLM = False
        

# ========== DB 연결/초기화 ==========
def get_conn():
    return psycopg2.connect(
        dbname=os.getenv("PGDATABASE", "mcpdb"),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", "postgres"),
        host=os.getenv("PGHOST", "localhost"),
        port=os.getenv("PGPORT", "5432"),
    )

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    # 개발자
    cur.execute("""
    CREATE TABLE IF NOT EXISTS developers (
        id SERIAL PRIMARY KEY,
        project VARCHAR(200),
        name VARCHAR(100) UNIQUE NOT NULL,
        experience_years INT DEFAULT 0,
        skills TEXT[] DEFAULT '{}',
        work_experiences TEXT DEFAULT '',
        workload INT DEFAULT 0
    );
    """)
    # 태스크 (임베딩 컬럼은 선택)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id SERIAL PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        requirements TEXT NOT NULL,
        required_skills TEXT[] DEFAULT '{}',
        assignee_id INT REFERENCES developers(id),
        project VARCHAR(200),
        status VARCHAR(20) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT NOW(),
        related_task_ids INT[] DEFAULT '{}',
        resolution_notes TEXT,
        resolution_method TEXT,
        related_db TEXT
    );
    """)
    # 가중치
    cur.execute("""
    CREATE TABLE IF NOT EXISTS assignment_weights (
        id INT PRIMARY KEY DEFAULT 1,
        w_similarity FLOAT DEFAULT 2.0,
        w_past_count FLOAT DEFAULT 1.5,
        w_workload FLOAT DEFAULT 1.2,
        w_skill FLOAT DEFAULT 1.0,
        w_project_bonus FLOAT DEFAULT 0.5,
        w_llm FLOAT DEFAULT 2.0
    );
    """)
    cur.execute("INSERT INTO assignment_weights (id) VALUES (1) ON CONFLICT DO NOTHING;")
    # trigram 확장(유사도)
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
    conn.commit()
    cur.close()
    conn.close()

init_db()

# ========== 유틸 ==========
def dict_rows(cur):
    cols = [desc[0] for desc in cur.description]
    for row in cur.fetchall():
        yield {col: row[i] for i, col in enumerate(cols)}

def safe_similarity(cur, text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    cur.execute("SELECT similarity(%s, %s)", (text_a, text_b))
    v = cur.fetchone()[0]
    return float(v or 0.0)

# ========== LLM 보정 ==========
def llm_rank_devs(task: Dict[str, Any], candidates: List[Dict[str, Any]], model: str) -> Dict[str, float]:
    """
    태스크와 후보 개발자 목록을 입력으로 받아,
    각 개발자 이름 -> 적합도(0~1) 점수 JSON을 반환하도록 LLM에 요청합니다.
    실패하면 빈 dict 반환.
    """
    if not USE_LLM or not candidates:
        return {}

    sys_prompt = (
        "당신은 '개발자 업무 분배 보조자'입니다. "
        "입력으로 주어진 태스크와 개발자 목록을 보고 각 개발자 적합도를 0.0~1.0 사이 float로 평가하세요. "
        "반드시 JSON만 출력: {\"이름\":0.92, ...}"
    )
    user_payload = {
        "task": task,
        "candidates": [
            {
                "name": d["name"],
                "project": d.get("project"),
                "experience_years": d.get("experience_years", 0),
                "skills": d.get("skills", []),
                "work_experiences": d.get("work_experiences", ""),
                "workload": d.get("workload", 0)
            } for d in candidates
        ]
    }
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=0.0
        )
        txt = resp.choices[0].message.content.strip()
        # JSON 영역만 추출
        s, e = txt.find("{"), txt.rfind("}")
        if s != -1 and e != -1:
            txt = txt[s:e+1]
        scores = json.loads(txt)
        out = {}
        for k, v in scores.items():
            try:
                out[k] = max(0.0, min(1.0, float(v)))
            except:
                pass
        return out
    except Exception as e:
        print("[WARN] LLM ranking failed:", e)
        return {}

# ========== 자동 배정 ==========
def auto_assign(conn, task_id: int):
    """
    분배 규칙:
    1) 과거 유사 TASK 경험(최대 유사도 + 유사 사례 수)
    2) 업무량 적은 순 (편차 클 시 저업무량 가중)
    3) 기술 매칭 수
    4) 프로젝트 일치 보너스
    5) (옵션) LLM 적합도 보정
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # 태스크/가중치/개발자 로드
    cur.execute("SELECT * FROM tasks WHERE id=%s", (task_id,))
    task = cur.fetchone()
    if not task:
        cur.close()
        return None, []

    cur.execute("SELECT * FROM assignment_weights WHERE id=1")
    W = dict(cur.fetchone())

    cur.execute("SELECT * FROM developers")
    devs = [dict(r) for r in cur.fetchall()]
    if not devs:
        cur.close()
        return None, []

    req_text = f"{task['title']} {task['requirements']}"
    required_skills = set(task["required_skills"] or [])
    project = task.get("project")

    # 후보 스코어
    scored = []
    workloads = []
    for d in devs:
        # 과거 최대 유사도
        cur.execute("""
          SELECT MAX(similarity(t.title || ' ' || t.requirements, %s))
          FROM tasks t WHERE t.assignee_id = %s
        """, (req_text, d["id"]))
        best_sim = float(cur.fetchone()[0] or 0.0)

        # 유사 사례 수
        cur.execute("""
          SELECT COUNT(*) FROM tasks t
          WHERE t.assignee_id = %s AND similarity(t.title || ' ' || t.requirements, %s) > 0.25
        """, (d["id"], req_text))
        past_cnt = int(cur.fetchone()[0])

        # 업무량 점수
        workload = int(d["workload"] or 0)
        workloads.append(workload)
        workload_score = 1.0 / (1 + workload)

        # 기술 매칭
        skill_match = len(required_skills & set(d["skills"] or []))

        # 프로젝트 보너스
        project_bonus = W["w_project_bonus"] if (project and d.get("project") == project) else 0.0

        base = (
            W["w_similarity"] * best_sim +
            W["w_past_count"] * (past_cnt / 5.0) +
            W["w_workload"] * workload_score +
            W["w_skill"] * skill_match +
            project_bonus
        )
        scored.append({"dev": d, "score": base, "workload": workload})

    # 업무량 편차 가중 (편차 5 이상이면 저업무량 보정)
    if scored:
        mx, mn = max(workloads), min(workloads)
        if mx - mn >= 5:
            for s in scored:
                # 약한 추가 보정
                s["score"] *= (1.10 if s["workload"] == mn else 1.0)

    # (옵션) LLM 보정
    if USE_LLM:
        llm_scores = llm_rank_devs(
            task={
                "title": task["title"],
                "requirements": task["requirements"],
                "required_skills": list(required_skills),
                "project": project
            },
            candidates=[x["dev"] for x in scored],
            model=OPENAI_MODEL
        )
        for s in scored:
            s["score"] += W["w_llm"] * llm_scores.get(s["dev"]["name"], 0.0)

    # 최고 득점자 선택
    scored.sort(key=lambda x: x["score"], reverse=True)
    winner = scored[0]["dev"]
    assignee_id = winner["id"]

    # 연관 과거 태스크(최근 50개 중 상위 5개)
    cur.execute("""
        SELECT id, title, requirements
        FROM tasks WHERE assignee_id=%s
        ORDER BY id DESC LIMIT 50
    """, (assignee_id,))
    rows = [dict(r) for r in cur.fetchall()]
    ranked = []
    for r in rows:
        sim = safe_similarity(cur, f"{r['title']} {r['requirements']}", req_text)
        if sim > 0:
            ranked.append((sim, r["id"]))
    ranked.sort(key=lambda t: t[0], reverse=True)
    related_ids = [tid for _, tid in ranked[:5]]

    # 반영
    cur.execute("UPDATE tasks SET assignee_id=%s, related_task_ids=%s WHERE id=%s",
                (assignee_id, related_ids, task_id))
    cur.execute("UPDATE developers SET workload = workload + 1 WHERE id=%s", (assignee_id,))
    conn.commit()
    cur.close()
    return assignee_id, related_ids

# ========== MCP 앱 ==========
app = FastMCP(name="DevAssignmentMCP")

@app.tool()
def seed_demo_data() -> dict:
    """
    [테스트용] 데모 개발자/태스크 데이터를 초기화합니다.
    시나리오:
      관리자가 처음 시스템을 세팅할 때 샘플 프로필과 과거 태스크를 만들어,
      자동 배정/조회/완료 처리 흐름을 즉시 시험할 수 있습니다.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM tasks;")
    cur.execute("DELETE FROM developers;")

    cur.execute("""
        INSERT INTO developers (project, name, experience_years, skills, work_experiences, workload) VALUES
        ('Alpha','김개발',6,ARRAY['python','fastapi','postgres','docker'],'백엔드/배치/DB튜닝',2),
        ('Beta','이개발',3,ARRAY['react','typescript','node','docker'],'프론트+간단 서버',1),
        ('Gamma','박개발',10,ARRAY['java','spring','postgres','kafka'],'대규모 트랜잭션',4),
        ('Delta','최개발',2,ARRAY['python','pandas','etl'],'데이터 가공/ETL',0)
    """)

    cur.execute("""
        INSERT INTO tasks (title, requirements, required_skills, assignee_id, project, status, resolution_notes, resolution_method, related_db)
        VALUES
        ('주문 집계 배치 수정','하루 주문 통계 배치 성능 최적화',ARRAY['python','postgres'],1,'Alpha','done','인덱스 추가','SQL 튜닝','orders'),
        ('관리자 UI 필터 추가','React 기반 상태/기간 필터링',ARRAY['react','typescript'],2,'Beta','done','쿼리 파라미터 설계','프론트 라우팅','adm'),
        ('고객 DB 마이그레이션','스키마 변경에 따른 데이터 이전',ARRAY['postgres','java'],3,'Gamma','done','검증 스크립트','ETL+검증','customers'),
        ('ETL 파이프라인 정리','매출 데이터 ETL 에러 핸들링 강화',ARRAY['python','etl'],4,'Delta','done','재시도/로그','모니터링','sales')
    """)
    conn.commit()
    cur.close()
    conn.close()
    return {"status": "ok"}

@app.tool()
def admin_add_developer(project: str, name: str, experience_years: int,
                        skills: List[str], work_experiences: str, workload: int = 0) -> dict:
    """
    [관리자] 개발자 정보를 등록합니다.
    시나리오:
      관리자는 신규 개발자를 추가하여 프로필(프로젝트/경력/보유기술/업무경험/현재 업무량)을 갱신합니다.
      이 정보는 이후 자동 배정 시 후보 선정과 랭킹에 활용됩니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
      INSERT INTO developers (project, name, experience_years, skills, work_experiences, workload)
      VALUES (%s,%s,%s,%s,%s,%s) RETURNING *;
    """, (project, name, experience_years, skills, work_experiences, workload))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return dict(row)

@app.tool()
def admin_add_task(title: str, requirements: str, required_skills: List[str], project: str = None) -> dict:
    """
    [관리자] 태스크를 등록합니다. 등록 즉시 자동 배정이 실행됩니다.
    시나리오:
      관리자가 신규 업무를 등록하면 시스템은
        - 과거 유사도(최대/건수)
        - 업무량
        - 기술 매칭
        - 프로젝트 일치 보너스
        - (옵션) LLM 적합도 보정
      을 종합해 담당 개발자를 자동 배정합니다.
      또한 개발자가 참고할 수 있도록 해당 담당자의 '연관 과거 태스크'를 함께 제공합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
      INSERT INTO tasks (title, requirements, required_skills, project, status)
      VALUES (%s,%s,%s,%s,'pending') RETURNING *;
    """, (title, requirements, required_skills, project))
    task = cur.fetchone()
    conn.commit()

    assignee_id, related_ids = auto_assign(conn, task["id"])
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM tasks WHERE id=%s", (task["id"],))
    updated = cur.fetchone()

    dev_name = None
    if assignee_id:
        cur.execute("SELECT name FROM developers WHERE id=%s", (assignee_id,))
        r = cur.fetchone()
        dev_name = r["name"] if r else None

    cur.close()
    conn.close()
    return {"task": dict(updated), "assignee_name": dev_name, "related_task_ids": related_ids}

@app.tool()
def dev_my_tasks(developer_name: str) -> dict:
    """
    [개발자] 본인에게 배정된 태스크 목록을 조회합니다.
    시나리오:
      개발자는 오늘 해야 할 업무 리스트를 확인하고, 각 업무마다 시스템이 추천한
      '연관 과거 태스크'를 참고해 빠르게 해결책(처리방법/관련DB/노하우)을 파악합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute("SELECT id FROM developers WHERE name=%s", (developer_name,))
    dev = cur.fetchone()
    if not dev:
        cur.close()
        conn.close()
        return {"tasks": []}

    cur.execute("""
      SELECT * FROM tasks
      WHERE assignee_id=%s AND status IN ('pending','in_progress')
      ORDER BY id DESC
    """, (dev["id"],))
    tasks = [dict(r) for r in cur.fetchall()]

    results = []
    for t in tasks:
        rel_ids = t.get("related_task_ids") or []
        related = []
        if rel_ids:
            cur.execute("SELECT id,title,requirements,status FROM tasks WHERE id = ANY(%s) ORDER BY id DESC",
                        (rel_ids,))
            related = [dict(r) for r in cur.fetchall()]
        results.append({"task": t, "related_tasks": related})

    cur.close()
    conn.close()
    return {"tasks": results}

@app.tool()
def dev_complete_task(task_id: int, resolution_notes: str, resolution_method: str, related_db: str) -> dict:
    """
    [개발자] 태스크를 완료로 변경하고 처리내역을 기록합니다.
    시나리오:
      개발자가 업무를 마감하면 처리 내용/방법/관련DB를 함께 저장합니다.
      이 기록은 이후 유사 업무 배정 시 '연관 태스크'로 전달되어 해결 시간 단축에 기여합니다.
      완료 시 담당자의 업무량은 1 감소합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute("SELECT assignee_id FROM tasks WHERE id=%s", (task_id,))
    r = cur.fetchone()
    if not r:
        cur.close()
        conn.close()
        return {"error": "task not found"}
    assignee_id = r["assignee_id"]

    cur.execute("""
      UPDATE tasks
         SET status='done',
             resolution_notes=%s,
             resolution_method=%s,
             related_db=%s
       WHERE id=%s
   RETURNING *;
    """, (resolution_notes, resolution_method, related_db, task_id))
    updated = cur.fetchone()

    if assignee_id:
        cur.execute("""
          UPDATE developers SET workload = GREATEST(workload - 1, 0)
          WHERE id=%s
        """, (assignee_id,))

    conn.commit()
    cur.close()
    conn.close()
    return {"task": dict(updated)}

@app.tool()
def search_similar_tasks(query: str, top_k: int = 5) -> dict:
    """
    [검색] 자연어 질의로 과거 유사 태스크를 조회합니다. (Postgres trigram 기반)
    시나리오:
      관리자가 새 업무를 등록하기 전에 비슷한 사례를 확인하거나,
      개발자가 레퍼런스를 빠르게 찾고 싶을 때 사용합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    cur.execute("SELECT id, title, requirements FROM tasks ORDER BY id DESC LIMIT 500")
    rows = [dict(r) for r in cur.fetchall()]
    scored = []
    for r in rows:
        s = safe_similarity(cur, f"{r['title']} {r['requirements']}", query)
        if s > 0:
            scored.append((s, r["id"], r["title"]))
    scored.sort(key=lambda t: t[0], reverse=True)
    result = [{"id": tid, "title": title, "score": round(score, 3)} for score, tid, title in scored[:top_k]]

    cur.close()
    conn.close()
    return {"results": result}

@app.tool()
def get_all_developers() -> dict:
    """
    [조회] 등록된 모든 개발자 목록을 조회합니다.
    시나리오:
      관리자가 전체 개발자 현황을 파악하거나,
      개발자 정보를 확인할 때 사용합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("SELECT * FROM developers ORDER BY id")
    developers = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return {"developers": developers}

@app.tool()
def get_all_tasks() -> dict:
    """
    [조회] 등록된 모든 업무(태스크) 목록을 조회합니다.
    시나리오:
      관리자가 전체 업무 현황을 파악하거나,
      업무 진행 상황을 확인할 때 사용합니다.
    """
    conn = get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute("""
        SELECT t.*, d.name as assignee_name, d.project as dev_project
        FROM tasks t 
        LEFT JOIN developers d ON t.assignee_id = d.id 
        ORDER BY t.id
    """)
    tasks = [dict(row) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return {"tasks": tasks}

@app.tool()
def admin_set_weights(w_similarity: float, w_past_count: float, w_workload: float,
                      w_skill: float, w_project_bonus: float, w_llm: float) -> dict:
    """
    [관리자] 자동 배정 알고리즘의 가중치를 설정합니다.
    시나리오:
      조직의 상황(업무량 편차, 기술 중요도, 도메인 연속성)에 따라 점수를 조정해
      원하는 배정 전략을 실시간으로 최적화합니다.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
      UPDATE assignment_weights
         SET w_similarity=%s,
             w_past_count=%s,
             w_workload=%s,
             w_skill=%s,
             w_project_bonus=%s,
             w_llm=%s
       WHERE id=1
    """, (w_similarity, w_past_count, w_workload, w_skill, w_project_bonus, w_llm))
    conn.commit()
    cur.close()
    conn.close()
    return {
        "status": "ok",
        "weights": {
            "similarity": w_similarity,
            "past_count": w_past_count,
            "workload": w_workload,
            "skill": w_skill,
            "project_bonus": w_project_bonus,
            "llm": w_llm
        }
    }

if __name__ == "__main__":
    # 표준 MCP (stdio) 서버 실행
    app.run()
