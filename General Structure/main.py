import argparse
import os
import config

# ###
import arabic_reshaper
from bidi.algorithm import get_display
def arabicformatter(text):
    reshaped_text = arabic_reshaper.reshape(text)   # fixes letter joining/shaping
    bidi_text = get_display(reshaped_text)           # fixes right-to-left order
    return bidi_text
# ###

def _check_indexes_exist():
    for path in (config.FAISS_INDEX_PATH, config.BM25_PATH, config.CHUNKS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Index files not found. Run `python main.py ingest` first."
            )

def _print_chunks(chunks):
    print("\nTop Retrieved Chunks:")
    i = 1
    for c in chunks:
        print(f" {i} - [{c['source']}] : score = {c['rerank_score']:.4f}  \nText Chunk: \n{c['text'][:100]} ...\n")
        i += 1


def _has_relevant_context(chunks) -> bool:
    if not chunks:
        return False
    return chunks[0]["rerank_score"] >= config.RELEVANCE_SCORE_THRESHOLD

def run_ingest():
    import ingest
    ingest.main()

def run_query(question: str):
    _check_indexes_exist()

    from retriever import HybridRetriever
    from llm import answer_query

    retriever = HybridRetriever()
    chunks = retriever.search(question)
    context_found = _has_relevant_context(chunks)

    if chunks:
        _print_chunks(chunks)
    else:
        print("No candidate chunks found — answering from general knowledge.")

    answer = answer_query(question, chunks, context_found=context_found)
    print("\n=== Answer ===")
    print(answer)

def run_chat():
    _check_indexes_exist()

    from retriever import HybridRetriever
    from llm import answer_query, translate_to_english

    print("Loading models ...")
    retriever = HybridRetriever()
    print("\nChatbot Ready! Ask in English/Arabic. ")
    print("Type 'exit' to quit.\n")

    while True:
        try: 
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting chat ...")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Exiting chat ...")
            break

        chunks = retriever.search(question)
        context_found = _has_relevant_context(chunks)

        if chunks:
            _print_chunks(chunks)
        else:
            print("No candidate chunks found — answering from general knowledge.")

        answer = answer_query(question, chunks, context_found=context_found)
        print(f"Context Found: {context_found}")
        print(f"\nAI: {arabicformatter(answer)}\n")

        AItranslation = translate_to_english(answer)
        print(f"AI Translator: {AItranslation}\n")

def main():
    parser = argparse.ArgumentParser(description="Arabic hybrid search + rerank RAG pipeline")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest", help="Build FAISS + BM25 indexes from documents/")

    query_parser = subparsers.add_parser("query", help="Ask a single one-off question")
    query_parser.add_argument("--q", required=True, help="The question to ask (English or Arabic)")

    subparsers.add_parser("chat", help="Start an interactive chat loop (default if no command given)")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest()
    elif args.command == "query":
        run_query(args.q)
    elif args.command == "chat" or args.command is None:
        run_chat()

if __name__ == "__main__":
    main()