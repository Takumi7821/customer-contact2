"""
このファイルは、画面表示以外の様々な関数定義のファイルです。
"""

############################################################
# ライブラリの読み込み
############################################################
import os
from dotenv import load_dotenv
import streamlit as st
import logging
import sys
import unicodedata
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import constants as ct


############################################################
# 設定関連
############################################################
load_dotenv()


############################################################
# 関数定義
############################################################

def build_error_message(message):
    """
    エラーメッセージと管理者問い合わせテンプレートの連結

    Args:
        message: 画面上に表示するエラーメッセージ

    Returns:
        エラーメッセージと管理者問い合わせテンプレートの連結テキスト
    """
    return "\n".join([message, ct.COMMON_ERROR_MESSAGE])


def create_rag_chain(db_name):
    """
    引数として渡されたDB内を参照するRAGのChainを作成

    Args:
        db_name: RAG化対象のデータを格納するデータベース名
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    docs_all = []
    # AIエージェント機能を使わない場合の処理
    if db_name == ct.DB_ALL_PATH:
        folders = os.listdir(ct.RAG_TOP_FOLDER_PATH)
        # 「data」フォルダ直下の各フォルダ名に対して処理
        for folder_path in folders:
            if folder_path.startswith("."):
                continue
            # フォルダ内の各ファイルのデータをリストに追加
            add_docs(f"{ct.RAG_TOP_FOLDER_PATH}/{folder_path}", docs_all)
    # AIエージェント機能を使う場合の処理
    else:
        # データベース名に対応した、RAG化対象のデータ群が格納されているフォルダパスを取得
        folder_path = ct.DB_NAMES[db_name]
        # フォルダ内の各ファイルのデータをリストに追加
        add_docs(folder_path, docs_all)

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    for doc in docs_all:
        doc.page_content = adjust_string(doc.page_content)
        for key in doc.metadata:
            doc.metadata[key] = adjust_string(doc.metadata[key])
    
    text_splitter = CharacterTextSplitter(
        chunk_size=ct.CHUNK_SIZE,
        chunk_overlap=ct.CHUNK_OVERLAP,
        separator="\n",
    )
    splitted_docs = text_splitter.split_documents(docs_all)

    embeddings = OpenAIEmbeddings()

    # すでに対象のデータベースが作成済みの場合は読み込み、未作成の場合は新規作成する
    if os.path.isdir(db_name):
        db = Chroma(persist_directory=".db", embedding_function=embeddings)
    else:
        db = Chroma.from_documents(splitted_docs, embedding=embeddings, persist_directory=".db")

    retriever = db.as_retriever(search_kwargs={"k": ct.TOP_K})

    question_generator_template = ct.SYSTEM_PROMPT_CREATE_INDEPENDENT_TEXT
    question_generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_generator_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_template = ct.SYSTEM_PROMPT_INQUIRY
    question_answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", question_answer_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        st.session_state.llm, retriever, question_generator_prompt
    )

    question_answer_chain = create_stuff_documents_chain(st.session_state.llm, question_answer_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


def add_docs(folder_path, docs_all):
    """
    フォルダ内のファイル一覧を取得

    Args:
        folder_path: フォルダのパス
        docs_all: 各ファイルデータを格納するリスト
    """
    files = os.listdir(folder_path)
    for file in files:
        # ファイルの拡張子を取得
        file_extension = os.path.splitext(file)[1]
        # 想定していたファイル形式の場合のみ読み込む
        if file_extension in ct.SUPPORTED_EXTENSIONS:
            # ファイルの拡張子に合ったdata loaderを使ってデータ読み込み
            loader = ct.SUPPORTED_EXTENSIONS[file_extension](f"{folder_path}/{file}")
        else:
            continue
        docs = loader.load()
        docs_all.extend(docs)


def run_company_doc_chain(param):
    """
    会社に関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # 会社に関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.company_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})
    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_service_doc_chain(param):
    """
    サービスに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値

    Returns:
        LLMからの回答
    """
    # サービスに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.service_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]

def run_customer_doc_chain(param):
    """
    顧客とのやり取りに関するデータ参照に特化したTool設定用の関数

    Args:
        param: ユーザー入力値
    
    Returns:
        LLMからの回答
    """
    # 顧客とのやり取りに関するデータ参照に特化したChainを実行してLLMからの回答取得
    ai_msg = st.session_state.customer_doc_chain.invoke({"input": param, "chat_history": st.session_state.chat_history})

    # 会話履歴への追加
    st.session_state.chat_history.extend([HumanMessage(content=param), AIMessage(content=ai_msg["answer"])])

    return ai_msg["answer"]


def delete_old_conversation_log(result):
    """
    古い会話履歴の削除

    Args:
        result: LLMからの回答
    """
    # LLMからの回答テキストのトークン数を取得
    response_tokens = len(st.session_state.enc.encode(result))
    # 過去の会話履歴の合計トークン数に加算
    st.session_state.total_tokens += response_tokens

    # トークン数が上限値を下回るまで、順に古い会話履歴を削除
    while st.session_state.total_tokens > ct.MAX_ALLOWED_TOKENS:
        # 最も古い会話履歴を削除
        removed_message = st.session_state.chat_history.pop(1)
        # 最も古い会話履歴のトークン数を取得
        removed_tokens = len(st.session_state.enc.encode(removed_message.content))
        # 過去の会話履歴の合計トークン数から、最も古い会話履歴のトークン数を引く
        st.session_state.total_tokens -= removed_tokens


def execute_agent_or_chain(chat_message):
    """
    AIエージェントもしくはAIエージェントなしのRAGのChainを実行

    Args:
        chat_message: ユーザーメッセージ
    
    Returns:
        LLMからの回答
    """
    logger = logging.getLogger(ct.LOGGER_NAME)

    # AIエージェント機能を利用する場合
    if st.session_state.agent_mode == ct.AI_AGENT_MODE_ON:
        # LLMによる回答をストリーミング出力するためのオブジェクトを用意
        st_callback = StreamlitCallbackHandler(st.container())
        # Agent Executorの実行（AIエージェント機能を使う場合は、Toolとして設定した関数内で会話履歴への追加処理を実施）
        result = st.session_state.agent_executor.invoke({"input": chat_message}, {"callbacks": [st_callback]})
        response = result["output"]
    # AIエージェントを利用しない場合
    else:
        # RAGのChainを実行
        result = st.session_state.rag_chain.invoke({"input": chat_message, "chat_history": st.session_state.chat_history})
        # 会話履歴への追加
        st.session_state.chat_history.extend([HumanMessage(content=chat_message), AIMessage(content=result["answer"])])
        response = result["answer"]

    # LLMから参照先のデータを基にした回答が行われた場合のみ、フィードバックボタンを表示
    if response != ct.NO_DOC_MATCH_MESSAGE:
        st.session_state.answer_flg = True
    
    return response


def adjust_string(s):
    """
    Windows環境でRAGが正常動作するよう調整
    
    Args:
        s: 調整を行う文字列
    
    Returns:
        調整を行った文字列
    """
    # 調整対象は文字列のみ
    if type(s) is not str:
        return s

    # OSがWindowsの場合、Unicode正規化と、cp932（Windows用の文字コード）で表現できない文字を除去
    if sys.platform.startswith("win"):
        s = unicodedata.normalize('NFC', s)
        s = s.encode("cp932", "ignore").decode("cp932")
        return s
    
    # OSがWindows以外の場合はそのまま返す
    return s