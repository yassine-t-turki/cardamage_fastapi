from openai import AzureOpenAI
from utils import load_api_keys

model_name = "test"
api_key = "ee6a8db755cf4ad5b9454d71c5c3d14f"
endpoint = "https://raneemsresource.openai.azure.com/"

api_keys = load_api_keys('keys.json')

model_deployment_name_azure = api_keys.get('model_deployment_name_azure')
endpoint_azure = api_keys.get('endpoint_azure')
api_key_azure = api_keys.get('api_key_azure')

class ChatSession:
    def __init__(self):
        self.azure_endpoint = endpoint_azure
        self.api_key = api_key_azure  # Replace with your OpenAI API key
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant working in the insurance sector. The user will give you a JSON output of a model that detected damages in a car. Your task is to guide the user throughout the necessary steps to complete a report. After recieving the metadata, make sure to recap the damages detected first"}
        ]
        self.client = AzureOpenAI(
                    azure_endpoint = self.azure_endpoint, 
                    api_key=self.api_key,
                    api_version="2024-02-01"
                    )
        
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})

    def run_request(self, user_message):
        self.add_message("user", user_message)

        # Make the API call with the entire conversation history
        response = self.client.chat.completions.create(
            model= model_deployment_name_azure,
            messages=self.messages
        )

        # Extract the assistant's reply
        assistant_message = response.choices[0].message.content

        # Add the assistant's message to the conversation history
        self.add_message("assistant", assistant_message)

        return assistant_message
