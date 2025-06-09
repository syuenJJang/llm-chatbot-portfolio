from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import anthropic
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from fastapi import UploadFile, File
import PyPDF2
from sentence_transformers import SentenceTransformer
import io
import uuid
import re
from typing import List, Dict

app = FastAPI()

templates = Jinja2Templates(directory="templates")

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# AWS 자격 증명 환경 변수에서 가져오기
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2")

awsauth = None
if aws_access_key and aws_secret_key:
    awsauth = AWS4Auth(aws_access_key, aws_secret_key, aws_region, 'aoss')
    print("AWS 자격 증명을 환경 변수에서 로드했습니다")
else:
    print("❌ AWS 자격 증명이 .env 파일에 설정되지 않았습니다")

opensearch_client = OpenSearch(
    hosts=[{
        'host': os.getenv("OPENSEARCH_HOST"),
        'port': 443
    }],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=False,
    connection_class=RequestsHttpConnection,
    timeout=60
)

def test_opensearch():
    """OpenSearch 연결 테스트"""
    if not awsauth:
        return False, "AWS 자격 증명 없음"
    
    try:
        # 403 오류를 피하기 위해 다른 API 시도
        indices = opensearch_client.cat.indices(format='json')
        return True, f"인덱스 수: {len(indices)}"
    except Exception as e:
        if "403" in str(e):
            return False, "권한 없음 - 데이터 액세스 정책 확인 필요"
        elif "404" in str(e):
            return False, "엔드포인트 문제"
        else:
            return False, str(e)

@app.on_event("startup")
async def startup_event():
    print("FastAPI 서버 시작!")
    
    # OpenSearch 연결 테스트
    success, message = test_opensearch()
    if success:
        print(f"OpenSearch 연결 성공: {message}")
    else:
        print(f"OpenSearch 연결 실패: {message}")

@app.get("/", response_class=HTMLResponse)
async def read_template(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat_anthropic(request: ChatRequest):
    try:
        response = client.messages.create(
            model = "claude-sonnet-4-20250514",
            max_tokens = 100,
            temperature = 0.7,
            messages = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": request.message
                }]
            }]
        )
        anthropic_response = response.content[0].text
        print(anthropic_response)
        
        return JSONResponse(content={
            "success": True,
            "response": anthropic_response
        })

    except Exception as e:
        print(e)
        return JSONResponse(content={
        "success": False,
        "error": "에러 발생" 
    })

def extract_keywords(query):
    """자연어 질문에서 핵심 키워드 추출"""
    
    # 정책 관련 핵심 키워드
    policy_keywords = [
        "연차", "월차", "휴가", "병가", "경조사", 
        "급여", "상여금", "보너스", "월급",
        "근무시간", "업무시간", "근로시간", "출퇴근",
        "유연근무", "재택근무", "휴게시간",
        "교육", "연수", "승진", "평가",
        "징계", "휴직", "퇴직", "해고"
    ]
    
    # 찾은 키워드들
    found_keywords = []
    query_lower = query.lower()
    
    for keyword in policy_keywords:
        if keyword in query_lower:
            found_keywords.append(keyword)
    
    # 키워드가 있으면 키워드 우선, 없으면 원본 쿼리
    if found_keywords:
        return " ".join(found_keywords)
    else:
        # 조사, 어미 제거한 간단한 버전
        cleaned = query.replace("은", "").replace("는", "").replace("이야", "").replace("야", "").replace("?", "").strip()
        return cleaned

@app.post("/rag-chat")
async def rag_chat(request: ChatRequest):
    try:
        original_query = request.message
        
        # 키워드 추출
        search_query = extract_keywords(original_query)
        print(f"원본 질문: {original_query}")
        print(f"검색 키워드: {search_query}")
        
        # 검색 쿼리 (search_query 사용)
        search_body = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "match_phrase": {
                                "content": {
                                    "query": search_query,
                                    "boost": 3.0
                                }
                            }
                        },
                        {
                            "match": {
                                "content": {
                                    "query": search_query,
                                    "boost": 2.0
                                }
                            }
                        },
                        {
                            "wildcard": {
                                "content": f"*{search_query}*"
                            }
                        },
                        # 유연한 매칭 추가
                        {
                            "multi_match": {
                                "query": search_query,
                                "fields": ["content"],
                                "type": "cross_fields",
                                "operator": "or",
                                "boost": 1.5
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": 3,
            "_source": ["content", "filename", "chunk_id", "chapter", "article"]
        }
        
        print(f"검색 쿼리: {search_body}")
        
        # 검색 실행
        search_response = opensearch_client.search(
            index="pdf-documents",
            body=search_body
        )
        
        print(f"검색 결과 개수: {len(search_response['hits']['hits'])}")
        
        # 결과 처리
        candidates = []
        for hit in search_response['hits']['hits']:
            source = hit['_source']
            candidates.append({
                "content": source['content'],
                "filename": source.get('filename', 'Unknown'),
                "chunk_id": source.get('chunk_id', 0),
                "chapter": source.get('chapter', 'N/A'),
                "article": source.get('article', 'N/A'),
                "score": hit['_score']
            })
        
        if not candidates:
            return JSONResponse(content={
                "success": False,
                "error": "관련 문서를 찾을 수 없습니다."
            })
        
        # 컨텍스트 구성
        context_parts = []
        for candidate in candidates:
            context_header = f"[{candidate['article']}]"
            context_parts.append(f"{context_header}\n{candidate['content']}")
        
        context = "\n\n".join(context_parts)
        print(f"컨텍스트 길이: {len(context)}")
        
        # Claude 프롬프트
        rag_prompt = f"""다음은 회사 정책 문서의 내용입니다. 이를 바탕으로 질문에 정확하게 답하세요.

        <정책 문서 내용>
        {context}
        </정책 문서 내용>

        질문: {original_query}

        답변 시 주의사항:
        1. 문서에 명시된 내용만을 바탕으로 답변하세요
        2. 구체적인 수치나 조건이 있다면 정확히 인용하세요
        3. 해당하는 조항을 언급하세요"""

        # Claude 응답 생성
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": rag_prompt}]
            }]
        )
        
        return JSONResponse(content={
            "success": True,
            "response": response.content[0].text,
            "sources": len(candidates),
            "search_type": "text_only"
        })
        
    except Exception as e:
        print(f"RAG 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(content={
            "success": False,
            "error": f"검색 오류: {str(e)}"
        })

@app.get("/test-opensearch")
async def test_opensearch_endpoint():
    """OpenSearch 연결 테스트 엔드포인트"""
    success, message = test_opensearch()
    return JSONResponse(content={
        "success": success,
        "message": message
    })

def smart_chunk_policy_document(text: str) -> List[Dict]:
    """정책 문서 구조를 고려한 스마트 청킹"""
    chunks = []
    current_chapter = ""
    current_article = ""
    
    # 텍스트 전처리
    text = re.sub(r'\s+', ' ', text)  # 여러 공백을 하나로
    text = text.replace('\n', ' ')
    
    # 장(章) 단위로 분할
    chapter_pattern = r'(제\d+장[^제]*?)(?=제\d+장|$)'
    chapters = re.findall(chapter_pattern, text, re.DOTALL)
    
    if not chapters:  # 장이 없으면 조 단위로 분할
        article_pattern = r'(제\d+조[^제]*?)(?=제\d+조|$)'
        articles = re.findall(article_pattern, text, re.DOTALL)
        
        if not articles:  # 조도 없으면 일반 청킹
            return simple_chunk(text)
        
        for i, article in enumerate(articles):
            if len(article.strip()) > 50:
                chunks.append({
                    "content": article.strip(),
                    "chapter": "미분류",
                    "article": f"제{i+1}조",
                    "type": "article"
                })
    else:
        for chapter in chapters:
            # 현재 장 제목 추출
            chapter_title_match = re.search(r'제\d+장[^\n]*', chapter)
            if chapter_title_match:
                current_chapter = chapter_title_match.group().strip()
            
            # 해당 장 내의 조 단위로 분할
            article_pattern = r'(제\d+조[^제]*?)(?=제\d+조|$)'
            articles = re.findall(article_pattern, chapter, re.DOTALL)
            
            if articles:
                for article in articles:
                    # 조 제목 추출
                    article_title_match = re.search(r'제\d+조[^\n]*', article)
                    if article_title_match:
                        current_article = article_title_match.group().strip()
                    
                    # 긴 조는 항목별로 세분화
                    if len(article) > 1000:
                        sub_items = split_by_items(article)
                        for j, item in enumerate(sub_items):
                            if len(item.strip()) > 100:
                                chunks.append({
                                    "content": item.strip(),
                                    "chapter": current_chapter,
                                    "article": current_article,
                                    "sub_item": j,
                                    "type": "sub_article"
                                })
                    else:
                        if len(article.strip()) > 50:
                            chunks.append({
                                "content": article.strip(),
                                "chapter": current_chapter,
                                "article": current_article,
                                "type": "article"
                            })
            else:
                # 조가 없으면 장 전체를 하나의 청크로
                if len(chapter.strip()) > 50:
                    chunks.append({
                        "content": chapter.strip(),
                        "chapter": current_chapter,
                        "article": "전체",
                        "type": "chapter"
                    })
    
    return chunks

def split_by_items(text: str) -> List[str]:
    """항목별로 텍스트 분할 (1., 2., 가., 나. 등)"""
    # 숫자 항목으로 분할
    items = re.split(r'\d+\.', text)
    if len(items) > 1:
        return [item.strip() for item in items if item.strip()]
    
    # 한글 항목으로 분할
    items = re.split(r'[가-힣]\.', text)
    if len(items) > 1:
        return [item.strip() for item in items if item.strip()]
    
    # 일반 청킹
    return [text[i:i+800] for i in range(0, len(text), 600)]

def simple_chunk(text: str) -> List[Dict]:
    """구조가 없는 경우 일반 청킹"""
    chunks = []
    chunk_size = 800
    overlap = 100
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append({
                "content": chunk.strip(),
                "chapter": "미분류",
                "article": f"청크{i//chunk_size + 1}",
                "type": "general"
            })
    
    return chunks

if __name__ == "__main__":
    import uvicorn
    success, message = test_opensearch()
    print(f"시작 전 연결 테스트: {message}")
    uvicorn.run(app, host="0.0.0.0", port=8000)