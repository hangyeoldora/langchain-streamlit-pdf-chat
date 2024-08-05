"""
    강의에 나온대로 패키지 설정하여 배포하면 에러 발생함
    langchain 0.2.x 이상
    langsmith 0.1.x 이상
"""
import streamlit as st
import re
from PyPDF2 import PdfReader
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import openai
from itertools import chain
from styles import css, user_template, ai_template


def init():
    # API 키 정보 로드
    # load_dotenv()
    st.set_page_config(page_title="1주차", page_icon=":sparkles:")
    # css 추가
    st.write(css, unsafe_allow_html=True)
    st.title(":sparkles: langchain pdf 챗봇 1주차 :sparkles:")

# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        # st.chat_message(chat_message.role).write(chat_message.content)
        if chat_message.role == "user":
            st.write(user_template.replace("{{MSG}}", chat_message.content), unsafe_allow_html=True)
        elif chat_message.role == "assistant":
            st.write(ai_template.replace("{{MSG}}", chat_message.content), unsafe_allow_html=True)

# 새로운 메세지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

def get_embedding_data(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
    result = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return result

# 체인 생성
def create_chain(prompt_type, api_key):
    # prompt | llm | output_parser
    if st.session_state["vector_data"] is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 친절한 AI 어시스턴트입니다. 다음의 질문에 간결하게 답변해 주세요."),
                ("user", "#Question:\n{question}"),
            ]
        )
    else:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 친절한 AI 어시스턴트입니다. 다음의 질문에 간결하게 답변해 주세요."),
                ("user", "#Question:\n{question}\n\n#Documents:\n{input_documents}"),
                ("assistant", "답변을 시작하세요.")
            ]
        )
    if prompt_type == "SNS 게시글":
        # windows 사용자만 cp949 mac은 없어도 되고 utf-8
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")
    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        openai_api_key=api_key
    )
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain


# 체인 생성
def create_audio_chain(api_key):
    # 오디오
    class WeatherAudioSummary(BaseModel):
        caster: str = Field(description="기상 소개를 하는 사람")
        subject: str = Field(description="제목으로 활용할 전체 내용 요약 짧은 한 줄")
        summary: str = Field(description="날씨 내용을 요약한 텍스트인데 4줄 안으로 요약")
        special_note: str = Field(description="특별히 강조를 해야 하는 부분")

    audio_prompt = PromptTemplate.from_template(
        """
            당신은 최고의 어시스턴트입니다. 다음 질문에 대해 인사없이 답변해주세요.

            #QUESTIONS
            아래 내용을 참고하는데 지역에 대한 설명과 강조해야할 부분은 넣어줘

            #SCRIPTS:
            {scripts}

            #FORMAT:
            {format}
        """
    )
    audio_output_parser = PydanticOutputParser(pydantic_object=WeatherAudioSummary)
    audio_prompt = audio_prompt.partial(format=audio_output_parser.get_format_instructions())
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        openai_api_key=api_key
    )
    audio_chain = audio_prompt | llm | audio_output_parser
    return audio_chain

# 사이드바 생성
def create_sidebar():
    # with는 컨테이너를 의미
    with st.sidebar:
        st.subheader("추가 기능")

        # 초기화 버튼 생성
        clear_btn = st.button("대화 초기화")

        # 초기화 버튼 누르면
        if clear_btn:
            st.session_state["messages"] = []
            st.session_state["vector_data"] = None
            st.session_state["pdf_uploaded"] = False

        # index는 기본값 (기본모드)
        selected_prompt = st.selectbox(
            "프롬프트를 선택해주세요",
            ("기본모드", "SNS 게시글", "요약"), index=0
        )

        pdf_data = st.file_uploader("pdf 파일을 업로드하세요.", type="pdf")

        # session에 pdf 업로드 여부를 체크하는 이유는 user_input 엔터 또는 보내기 클릭 시,
        # 페이지가 재랜더링되어서 업로드된 pdf가 다시 임베딩 과정을 거침
        if pdf_data is not None and not st.session_state.get("pdf_uploaded", False):
            st.session_state["pdf_uploaded"] = True
            with st.spinner("pdf 처리중"):
                try:
                    pdf_reader = PdfReader(pdf_data)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    st.session_state["vector_data"] = get_embedding_data(text)
                    # st.success 추가 수정 작업 필요 -> spinner 처리 후 등장했다가 2초 뒤에 사라지게 하기.
                    st.success("파일 처리 성공!")
                except Exception as e:
                    st.error(f"pdf 처리 중 에러 발생: {e}")

        return selected_prompt

# 답변 텍스트 정제
def format_text(text):
    # 줄바꿈 및 HTML 공백 처리
    text = text.replace('\n', '<br>').replace(' ', '&nbsp;')
    # 마침표 다음에 줄 바꿈을 추가하는데ㅔ 숫자 이외의 문자에 대해 정규식 적용했으나 수정 필요..
    formatted_text = re.sub(r'(?<!\d)\.(?!\d)', '.<br>', text)
    return formatted_text

def main():
    # 기본 세팅
    init()

    if "api_key" not in st.session_state:
        st.session_state["api_key"] = None

    if st.session_state["api_key"] is None:
        st.subheader("OpenAI API Key를 입력하세요.")
        api_key = st.text_input("OpenAI API Key", type="password")
        if st.button('확인'):
            if api_key:
                st.session_state["api_key"] = api_key
                st.session_state["messages"] = []
                st.session_state["vector_data"] = None
                st.session_state["show_main_page"] = True
                st.rerun()
            else:
                st.error("api key를 입력해주세요.")
        return

    if st.session_state.get('show_main_page', False):
        # 처음 1번만 실행하기 위한 코드
        if "messages" not in st.session_state:
            # 대화기록을 저장하기 위한 용도로 생성되며 새로고침되도 안 사라짐
            st.session_state["messages"] = []

        audio_chain = create_audio_chain(st.session_state["api_key"])
        # 사이드바 생성
        selected_prompt = create_sidebar()
        audio_file = st.sidebar.file_uploader("업로드 오디오", type=["wav", "mp3", "m4a"])
        if st.sidebar.button('transcribe audio'):
            if audio_file is not None:
                print(audio_file)
                st.sidebar.success('Transcribe Audio')
                openai.api_key = st.session_state["api_key"]
                transcription = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                st.sidebar.success('Transcription complete')
                st.markdown(transcription.text)
                response = audio_chain.invoke({"scripts": transcription.text})
                print(response)
                response.summary = response.summary.replace('. ', '.\n')
                response.special_note = response.special_note.replace('. ', '.\n')

                output = (
                    f"caster='{response.caster}'\n"
                    f"subject='{response.subject}'\n"
                    f"summary='{response.summary}'\n"
                    f"special_note='{response.special_note}'"
                )
                st.markdown(f"```text\n{output}\n```")
            else:
                st.sidebar.error("please upload an audio file")
        # 이전 대화 기록 출력
        print_messages()

        # 사용자의 입력
        user_input = st.chat_input("궁금한 내용을 물어보세요!")

        # 만약 사용자 입력이 들어오면...(엔터)
        if user_input:
            # st.write(f"사용자 입력: {user_input}")
            # 사용자의 입력
            st.write(user_template.replace("{{MSG}}", user_input), unsafe_allow_html=True)
            # chain 생성
            chain = create_chain(selected_prompt, st.session_state["api_key"])

            if st.session_state["vector_data"] is not None:
                faiss_db = st.session_state["vector_data"]

                """
                    faiss 질의 참조
                    "https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/#as-a-retriever"
                    
                    faiss에서 쿼리에 대한 검색 방법은 크게 두 가지
                    1. 표준적인 방법인 'similarity_search' 사용하기
                    2. 벡터스토어를 Retriever클래스로 변환하는 방법 (as_retriever())
                """
                # retriever 방법은 여러 번 사용해봤기 때문에 similarity_search 적용
                docs = faiss_db.similarity_search(user_input)
                print(docs)
                response = chain.stream({"question": user_input, "input_documents": docs})
            else:
                response = chain.stream({"question": user_input})

            # html template 내의 MSG 값만 변경해야하기 때문에 container 생성
            with st.container():
                container = st.empty()
                ai_answer = ""

                for token in response:
                    ai_answer += token
                    formatted_message = format_text(ai_answer)
                    full_html = ai_template.replace("{{MSG}}", formatted_message)
                    container.write(full_html, unsafe_allow_html=True)

            """
                유저가 질문 던짐 -> with로 열면 ai 채팅 공간 껍데기를 만듬
                st.empty 하는 순간 공간(메시지 입력할 타겟) 생성
                아래는 기존 테디 강의에 나온 답변 stream 처리 코드
            """
            # with st.chat_message("assistant"):
            #     # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력
            #     container = st.empty()
            #     ai_answer = ""
            #     for token in response:
            #         ai_answer += token
            #         container.markdown(ai_answer)

            add_message("user", user_input)
            add_message("assistant", ai_answer)

if __name__ == '__main__':
    main()
