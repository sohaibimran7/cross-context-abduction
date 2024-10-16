import pytest
from src.finetuners import OpenAIFinetuner

def test_prepare_training_data():
    finetuner = OpenAIFinetuner()
    
    # Sample input log
    input_log = '''
    {"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there! How can I assist you today?"}]}
    {"messages": [{"role": "user", "content": "What's the weather like?"}, {"role": "assistant", "content": "I'm sorry, but I don't have access to real-time weather information. You would need to check a weather service or app for current conditions."}]}
    '''
    
    # Prepare training data
    training_data = finetuner._prepare_training_data(input_log)
    
    # Assertions
    assert len(training_data) == 2, "Expected 2 valid training examples"
    
    # Check the first example
    assert training_data[0]["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]
    
    # Check the second example
    assert training_data[1]["messages"] == [
        {"role": "system", "content": " "}, # Empty system message with a space expected where none was provided
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'm sorry, but I don't have access to real-time weather information. You would need to check a weather service or app for current conditions."}
    ]

    no_system_finetuner = OpenAIFinetuner(msg_roles_to_extract=["user", "assistant"])

    no_system_training_data = no_system_finetuner._prepare_training_data(input_log)

    assert len(no_system_training_data) == 2, "Expected 2 valid training examples"

    assert no_system_training_data[0]["messages"] == [
        {"role": "system", "content": " "}, # Empty system message with a space expected where none was provided
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"}
    ]

    assert no_system_training_data[1]["messages"] == [
        {"role": "system", "content": " "}, # Empty system message with a space expected where none was provided
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I'm sorry, but I don't have access to real-time weather information. You would need to check a weather service or app for current conditions."}
    ]
    
    




