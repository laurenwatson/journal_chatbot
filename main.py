def respond(prompt):
    return "Okay!"

if __name__=='__main__':

    while True:
        user_input = input("Lauren: ")
        if user_input.lower() in ["quit"]:
            break

        response = respond(user_input)
        print("Journal: ", response)
