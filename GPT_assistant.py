from openai import OpenAI
import os
import json

KEPT_QUERY = ""
LAST_THREAD_ID = None

client = OpenAI(api_key="")

#assistant 1 to generate the users.
user_assistant = client.beta.assistants.retrieve("asst_2FoxMbK2Ql1FZmvYY9mjZ4XE")

data_assistant = client.beta.assistants.retrieve("asst_wRWUY9bDkJCSOiZG4pmMwWCu")


def generate_users():

    for i in range(1, 9):
        f = open(f"data/users/{i}.json", "a")
        my_thread = client.beta.threads.create()
        for i in range(25):
            my_message = client.beta.threads.messages.create(
                thread_id=my_thread.id,
                role='user',
                content="generate one user"
            )
            my_run = client.beta.threads.runs.create(
                thread_id=my_thread.id,
                assistant_id=user_assistant.id,
            )
            print(f"This is the run object: {my_run}")
            while my_run.status in ["queued", "running"]:
                keep_retrieving_run = client.beta.threads.runs.retrieve(
                    thread_id=my_thread.id,
                    run_id=my_run.id,
                )
                print(f"this is the run object: {keep_retrieving_run.status} \n")
                if keep_retrieving_run.status == "queued" or keep_retrieving_run.status == "in_progress":
                    pass
                else:
                    print(f"Run status: {keep_retrieving_run.status}")
                    break
            if keep_retrieving_run.status == "completed":
                print("\n")

                # Step 6: Retrieve the Messages added by the Assistant to the Thread
                all_messages = client.beta.threads.messages.list(
                    thread_id=my_thread.id
                )

                print("------------------------------------------------------------ \n")

                print(f"User: {my_message.content[0].text.value}")
                print(f"Assistant: {all_messages.data[0].content[0].text.value}")
                response = all_messages.data[0].content[0].text.value
                resp = response.strip()
            f.write(all_messages.data[0].content[0].text.value.strip())
        f.close()


#generate_users()

def generate_journals():
    for i in range(1, 9):
        f = open(f"data/journals/{i}.json", "a")
        with open(f'data/users/{i}.json') as users:
            data = json.load(users)
        my_thread = client.beta.threads.create()
        for item in data:

            my_message = client.beta.threads.messages.create(
                thread_id=my_thread.id,
                role="user",
                content=str(item)
            )
            my_run = client.beta.threads.runs.create(
                thread_id=my_thread.id,
                assistant_id=data_assistant.id
            )
            while my_run.status in ["queued", "running"]:
                keep_retrieving_run = client.beta.threads.runs.retrieve(
                    thread_id=my_thread.id,
                    run_id=my_run.id,
                )
                print(f"this is the run object: {keep_retrieving_run.status} \n")
                if keep_retrieving_run.status == "queued" or keep_retrieving_run.status == "in_progress":
                    pass
                else:
                    #print(f"Run status: {keep_retrieving_run.status}")
                    break
            if keep_retrieving_run.status == "completed":
                print("\n")

                # Step 6: Retrieve the Messages added by the Assistant to the Thread
                all_messages = client.beta.threads.messages.list(
                    thread_id=my_thread.id
                )

                print("------------------------------------------------------------ \n")

                print(f"Assistant: {all_messages.data[0].content[0].text.value}")
                response = all_messages.data[0].content[0].text.value
                resp = response.strip()
            f.write(all_messages.data[0].content[0].text.value.strip())
        f.close()


generate_journals()
