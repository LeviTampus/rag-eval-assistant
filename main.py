from generate import answer


def main():
    print("Anime Lore Assistant — type 'quit' to exit.")

    while True:
        question = input("\nYou: ").strip()
        if question == "" or question.strip() == "quit":
            print("Goodbye")
            break
        try:
            result = answer(question)
            print(result["answer"])
            print("Sources:", ", ".join(result["sources"]))
        except Exception as e:
            print("Sorry, something went wrong — please try again.")


if __name__ == "__main__":
    main()